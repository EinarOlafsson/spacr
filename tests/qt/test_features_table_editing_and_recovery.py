"""Keep the displayed field, its files and scan failures consistent."""
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent

from spacr.qt.widgets import measure_input_table as module


@pytest.fixture
def table(qtbot):
    widget = module.MeasureInputTable(threaded=False)
    qtbot.addWidget(widget)
    return widget


def test_sorted_browse_and_remove_reach_the_visible_field(table):
    first = table.add_field()
    second = table.add_field()
    table._grid.sortItems(0, Qt.DescendingOrder)
    assert table._grid.item(0, 0).text() == second.label
    table.set_file_picker(lambda caption: "/local/chosen.tif")
    table._grid.cellDoubleClicked.emit(0, 3)
    assert second.channels == {0: "/local/chosen.tif"}
    assert first.channels == {}
    table._grid.selectRow(0)
    assert table.remove_selected() == 1
    assert table.table().rows == [first]
    assert table.remove_selected() == 0


@pytest.mark.xfail(strict=True, raises=RuntimeError,
    reason="288: editing rebuilds/deletes SortableTableItem during setData; reported in 325")
def test_editing_a_sorted_well_finishes_without_deleting_the_active_item(table):
    first = table.add_field()
    second = table.add_field()
    table._grid.sortItems(0, Qt.DescendingOrder)
    table._grid.item(0, 1).setText(" B03 ")
    assert second.well == "B03" and first.well == "A01"


def test_cancelled_picker_and_identity_cells_preserve_assignments(table):
    row = table.add_field()
    assert table.assign_file(0, "channel:0", "/local/original.tif")
    asked = []
    table.set_file_picker(lambda caption: asked.append(caption) or "")
    table._grid.cellDoubleClicked.emit(0, 0)
    assert asked == []
    table._grid.cellDoubleClicked.emit(0, 3)
    assert len(asked) == 1
    assert row.channels == {0: "/local/original.tif"}


@pytest.mark.parametrize("row,column", [(-1, "cell"), (1, "cell"),
    (0, "channel:-1"), (0, "channel:2"), (0, "pathogen"), (0, "unknown")])
def test_invalid_destination_never_remembers_or_assigns_file(table, row, column):
    field = table.add_field()
    assert not table.assign_file(row, column, "/local/rejected.tif")
    assert field.channels == field.masks == {}
    assert table._known_paths == []


def test_shape_controls_remove_only_files_from_removed_columns(table):
    row = table.add_field()
    table.assign_file(0, "channel:0", "/local/c1.tif")
    table.assign_file(0, "channel:1", "/local/c2.tif")
    table.assign_file(0, "cell", "/local/cell.tif")
    table._role_boxes["nucleus"].setChecked(True)
    table.assign_file(0, "nucleus", "/local/nucleus.tif")
    table._channels.setValue(1)
    table._role_boxes["nucleus"].setChecked(False)
    assert row.channels == {0: "/local/c1.tif"}
    assert row.masks == {"cell": "/local/cell.tif"}
    table._plate.setText("experiment")
    assert table.table().plate == "experiment"
    assert "experiment" in table._grid.item(0, 0).toolTip()
    table._plate.clear()
    assert table.table().plate == "drawn"


def test_clear_cancels_pending_scan_and_forgets_all_file_memory(table, monkeypatch):
    cancelled = []
    monkeypatch.setattr(table._scanner, "cancel", lambda: cancelled.append(True))
    table.add_paths(["/local/fov001_C1.tif", "/local/unmatched.tif"])
    assert table.unassigned()
    table._clear.click()
    assert cancelled == [True]
    assert table.table().rows == []
    assert table.table().channel_tokens == ()
    assert table._known_paths == [] and table.unassigned() == []
    assert table.reapply_regex() == 0


def test_scan_submission_refusal_falls_back_to_assigning_file_names(table, monkeypatch):
    submitted = []
    monkeypatch.setattr(table._scanner, "submit",
                        lambda *args: submitted.append(args) or False)
    table.add_dropped([])
    assert submitted == []
    table.add_dropped(["/local/fov001_C1.tif"])
    assert len(submitted) == 1
    assert table.table().rows[0].channels == {0: "/local/fov001_C1.tif"}
    assert not table.is_scanning()


def test_failed_scan_reports_error_without_destroying_existing_rows(table):
    row = table.add_field()
    table._scan_failed("permission denied")
    assert "permission denied" in table._regex_status.text()
    assert table.table().rows == [row]
    table._files_found(None)
    assert table.table().rows == [row]


def test_invalid_regex_application_preserves_current_assignments(table):
    table.add_paths(["/local/fov001_C1.tif"])
    row = table.table().rows[0]
    table._regex.setText("(?P<field>")
    assert table.add_paths(["/local/fov002_C1.tif"]) == 0
    assert "will not compile" in table._regex_status.text()
    assert table.table().rows == [row]
    assert row.channels == {0: "/local/fov001_C1.tif"}


def test_remote_urls_and_text_are_not_accepted_as_local_files(table):
    remote, text = QMimeData(), QMimeData()
    remote.setUrls([QUrl("https://example.invalid/image.tif")])
    text.setText("/local/not-a-file-drop.tif")
    for mime in (remote, text):
        enter = QDragEnterEvent(QPoint(2, 2), Qt.CopyAction, mime,
                                Qt.LeftButton, Qt.NoModifier)
        move = QDragMoveEvent(QPoint(2, 2), Qt.CopyAction, mime,
                              Qt.LeftButton, Qt.NoModifier)
        drop = QDropEvent(QPointF(2, 2), Qt.CopyAction, mime,
                          Qt.LeftButton, Qt.NoModifier)
        table.dragEnterEvent(enter)
        table.dragMoveEvent(move)
        table.dropEvent(drop)
        assert not enter.isAccepted() and not move.isAccepted() and not drop.isAccepted()
    assert module._dropped_paths(SimpleNamespace()) == []
    assert table.table().rows == []


def test_walk_expands_only_one_level_and_reports_io_failure(tmp_path, monkeypatch):
    folder = tmp_path / "fields"
    folder.mkdir()
    (folder / "nested").mkdir()
    (folder / "nested" / "hidden.tif").touch()
    (folder / "b.tif").touch()
    (folder / "a.tif").touch()
    assert module._walk([str(folder)]) == (
        [str(folder / "a.tif"), str(folder / "b.tif")], "")
    def fail(_paths):
        raise PermissionError("denied")
    monkeypatch.setattr(module, "files_under", fail)
    assert module._walk([str(folder)]) == ([], "denied")
