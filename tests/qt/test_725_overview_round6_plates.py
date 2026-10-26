"""Overview round 6 plate items: several lines collapse at once, the Add plates
picker opens big enough and resizes by its edges, and each plate may keep
its own filename regex.

Item 725, 1.5.1.6, from the maintainer's round 6 notes.
"""
import json
import threading
from pathlib import Path

import numpy as np
import pytest
import tifffile

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QMenu, QPushButton

from spacr.qt.widgets import qc_field_browser as browser


PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"
OTHER = r"(?P<plateID>q\d+)_(?P<wellID>[A-Z]\d+)_s(?P<fieldID>\d+)_w(?P<chanID>\d+)\.tif"


def _plate(folder, wells, name="p1-{well}-f1-c{channel}.tif"):
    folder.mkdir(parents=True, exist_ok=True)
    for index, well in enumerate(wells):
        for channel in (1, 2):
            pixels = np.full((24, 32), 100 * (index + 1) + channel, dtype=np.uint16)
            tifffile.imwrite(folder / name.format(well=well, channel=channel), pixels)
    return folder


def _runners(view):
    return (view._index_jobs, view._image_jobs, view._thumbnail_jobs, view._field_thumbnail_jobs,
            view._montage_jobs, view._export_jobs, view._metadata_jobs, view._measurement_jobs,
            view._native_jobs, view._image_stack_jobs, view._group_jobs)


def _retire(view, qtbot):
    view._cancel()
    qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in _runners(view)), timeout=15000)
    for runner in _runners(view):
        runner.shutdown(timeout_ms=3000)
    view.close()


def _open(qtbot, source, **settings):
    browser._OVERVIEW_PLANE_CACHE.clear()
    browser._OVERVIEW_INFO_CACHE.clear()
    view = browser._FieldOverview(settings={
        "source": source, "source_scope": "folder", "custom_regex": PATTERN,
        "channels": (1, 2), "geometry": (3, 3), "representatives": 1, **settings})
    qtbot.addWidget(view, before_close_func=lambda widget: _retire(widget, qtbot))
    view.resize(1500, 1000)
    view.show()
    view._scan()
    qtbot.waitUntil(lambda: bool(view._fields) and view._normalization_ready
                    and not view._index_jobs.is_busy(), timeout=15000)
    return view


@pytest.fixture
def view(qtbot, tmp_path):
    return _open(qtbot, str(_plate(tmp_path / "plateA", ("A01", "A02", "B01", "B02", "C01"))))


def _select_rows(view, rows):
    view.wells.clearSelection()
    for row in rows:
        for column in range(view.wells.columnCount()):
            item = view.wells.item(row, column)
            if item is not None:
                item.setSelected(True)


def _captions(view, axis, index):
    menu = QMenu()
    view._line_actions(menu, axis, index)
    return menu, [action.text() for action in menu.actions()]


def test_collapse_from_the_menu_takes_every_selected_row(view):
    _select_rows(view, (0, 1))
    menu, captions = _captions(view, "row", 0)
    names = ", ".join(view._line_name("row", row) for row in (0, 1))
    assert captions == [f"Collapse rows {names}"]
    menu.actions()[0].trigger()
    assert view._collapsed_lines()[0] == {0, 1}
    menu, captions = _captions(view, "row", 1)
    assert captions == [f"Show rows {names}"]
    menu.actions()[0].trigger()
    assert view._collapsed_lines()[0] == set()


def test_a_line_outside_the_selection_collapses_alone(view):
    _select_rows(view, (0, 1))
    menu, captions = _captions(view, "row", 2)
    assert captions == [f"Collapse row {view._line_name('row', 2)}"]
    menu.actions()[0].trigger()
    assert view._collapsed_lines()[0] == {2}


def test_columns_collapse_together_too(view):
    view.wells.clearSelection()
    for column in (0, 1):
        view.wells.item(0, column).setSelected(True)
    menu, captions = _captions(view, "column", 1)
    names = ", ".join(view._line_name("column", column) for column in (0, 1))
    assert captions == [f"Collapse columns {names}"]
    menu.actions()[0].trigger()
    assert view._collapsed_lines()[1] == {0, 1}


def test_a_header_double_click_collapses_the_whole_selection(view, qtbot):
    _select_rows(view, (0, 1))
    header = view.wells.verticalHeader()
    point = QPoint(header.width() // 2, header.sectionViewportPosition(0) + header.sectionSize(0) // 2)
    qtbot.mouseDClick(header.viewport(), Qt.LeftButton, pos=point)
    assert view._collapsed_lines()[0] == {0, 1}


def _drag(dialog, edge, delta):
    start = dialog.mapToGlobal(edge)
    QApplication.sendEvent(dialog, QMouseEvent(QEvent.MouseButtonPress, edge, start, Qt.LeftButton,
                                               Qt.LeftButton, Qt.NoModifier))
    QApplication.sendEvent(dialog, QMouseEvent(QEvent.MouseMove, edge + delta, start + delta, Qt.NoButton,
                                               Qt.LeftButton, Qt.NoModifier))
    QApplication.sendEvent(dialog, QMouseEvent(QEvent.MouseButtonRelease, edge + delta, start + delta,
                                               Qt.LeftButton, Qt.NoButton, Qt.NoModifier))


def test_the_add_plates_picker_opens_big_with_every_button_shown_and_resizes_by_its_edges(view, monkeypatch):
    monkeypatch.setattr(QFileDialog, "exec", lambda dialog: 0)
    assert view._choose_plates() == []
    dialog = view._add_plates_dialog
    assert dialog.width() >= 720 and dialog.height() >= 480
    assert dialog.width() >= dialog.minimumSizeHint().width()
    assert dialog.height() >= dialog.minimumSizeHint().height()
    assert getattr(dialog, "_spacr_resizer", None) is not None
    dialog.show()
    QApplication.processEvents()
    buttons = [button for button in dialog.findChildren(QPushButton) if button.isVisible()]
    assert buttons
    for button in buttons:
        assert dialog.rect().contains(button.mapTo(dialog, QPoint(0, 0)))
        assert dialog.rect().contains(button.mapTo(dialog, QPoint(button.width() - 1, button.height() - 1)))
    width = dialog.width()
    _drag(dialog, QPointF(dialog.width() - 3, dialog.height() / 2), QPointF(-60, 0))
    assert dialog.width() == width - 60
    dialog.close()


def test_the_wizard_add_folders_picker_is_the_same_picker(qtbot, tmp_path, monkeypatch):
    _plate(tmp_path / "plate1", ("A01",))
    wizard = browser._OverviewMetadataWizard(str(tmp_path / "plate1"), {"custom_regex": PATTERN})
    wizard._startup_timer.stop()
    qtbot.addWidget(wizard)
    monkeypatch.setattr(QFileDialog, "exec", lambda dialog: 0)
    assert wizard._choose_more_sources() == []
    dialog = wizard._add_folders_dialog
    assert dialog.width() >= 720 and dialog.height() >= 480
    assert getattr(dialog, "_spacr_resizer", None) is not None
