"""Exercise the grid's resize gesture and help failures without a browser."""
from types import SimpleNamespace

import pytest
from PySide6 import QtGui
from PySide6.QtCore import QEvent, QModelIndex, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from spacr.qt.widgets import object_settings_grid as module


@pytest.fixture
def grid(qtbot, qt_theme_applied):
    widget = module.ObjectSettingsGrid()
    qtbot.addWidget(widget)
    widget.set_settings({"cell_channel": 0, "nucleus_channel": None,
                         "cell_diameter": 20, "nucleus_diameter": 10,
                         "cell_model_name": "cell", "nucleus_model_name": "nucleus"})
    widget.resize(700, 550)
    widget.show()
    qtbot.waitUntil(lambda: widget._grip.width() > 100)
    return widget


def mouse(widget, kind, global_y, button, buttons):
    event = QMouseEvent(kind, QPointF(4, 3), QPointF(30, global_y),
                        button, buttons, Qt.NoModifier)
    QApplication.sendEvent(widget, event)


def test_drag_resizes_from_press_and_double_click_restores_content_height(grid):
    grip = grid._grip
    start = grid._table.height()
    mouse(grip, QEvent.MouseButtonPress, 100, Qt.LeftButton, Qt.LeftButton)
    mouse(grip, QEvent.MouseMove, 190, Qt.NoButton, Qt.LeftButton)
    assert grid._table.height() == start + 90
    mouse(grip, QEvent.MouseMove, 210, Qt.NoButton, Qt.LeftButton)
    assert grid._table.height() == start + 110
    mouse(grip, QEvent.MouseMove, -1000, Qt.NoButton, Qt.LeftButton)
    assert grid._table.height() == grid.MIN_TABLE_H
    mouse(grip, QEvent.MouseButtonRelease, -1000, Qt.LeftButton, Qt.NoButton)
    assert grip._press_y is None
    mouse(grip, QEvent.MouseMove, 300, Qt.NoButton, Qt.NoButton)
    assert grid._table.height() == grid.MIN_TABLE_H
    mouse(grip, QEvent.MouseButtonDblClick, 300, Qt.LeftButton, Qt.LeftButton)
    assert grid._user_height is None
    assert grid._table.height() == start
    assert grip.sizeHint().height() == grip.HEIGHT


def test_right_drag_and_double_click_do_not_change_chosen_height(grid):
    grid.set_user_height(260)
    mouse(grid._grip, QEvent.MouseButtonPress, 100, Qt.RightButton, Qt.RightButton)
    mouse(grid._grip, QEvent.MouseMove, 200, Qt.NoButton, Qt.RightButton)
    mouse(grid._grip, QEvent.MouseButtonDblClick, 200, Qt.RightButton, Qt.RightButton)
    assert grid._table.height() == 260
    assert grid._grip._press_y is None


def test_missing_cells_are_not_editable_or_silently_added(qtbot):
    model = module.ObjectSettingsModel()
    model.set_table({"channel": {"cell": None},
                     "diameter": {"cell": 20, "nucleus": 10}})
    absent = model.index(0, model.objects().index("nucleus"))
    before = model.table()
    assert not model.flags(absent) & Qt.ItemIsEditable
    assert model.data(absent) is None
    assert "does not ask" in model.data(absent, Qt.ToolTipRole)
    assert not model.setData(absent, "2")
    assert model.table() == before
    invalid = QModelIndex()
    assert model.data(invalid) is None
    assert not model.flags(invalid) & Qt.ItemIsEditable
    assert not model.setData(invalid, "2")
    valid = model.index(1, 0)
    assert not model.setData(valid, "2", Qt.DisplayRole)
    assert model.data(valid, Qt.DecorationRole) is None
    assert model.headerData(0, Qt.Vertical, Qt.ToolTipRole) is None
    assert model.headerData(0, Qt.Horizontal, Qt.ToolTipRole)
    assert model.headerData(0, Qt.Horizontal, Qt.DecorationRole) is None


@pytest.mark.parametrize("stored,typed,expected", [
    (False, "yes", True), (True, "off", None),
    (None, "false", False), (None, "true", True),
    (None, "custom checkpoint", "custom checkpoint"),
    (10, "invalid numeric input", "invalid numeric input"),
])
def test_editor_preserves_explicit_typed_values(stored, typed, expected):
    model = module.ObjectSettingsModel()
    model.set_table({"custom": {"cell": stored, "nucleus": stored}})
    assert model.setData(model.index(0, 0), typed)
    actual = model.table()["custom"][model.objects()[0]]
    assert actual == expected and type(actual) is type(expected)


def test_leaving_cells_cancels_pending_help_and_entering_band_keeps_it(grid):
    proxy = grid._table.model()
    index = proxy.mapFromSource(grid._model.index(0, 0))
    pos = grid._table.visualRect(index).center()
    grid._offer_tooltip(pos)
    assert grid._help_show_timer.isActive()
    grid._offer_tooltip(QPoint(-20, -20))
    assert not grid._help_show_timer.isActive()
    assert grid._help_hide_timer.isActive()
    grid.eventFilter(grid._help_band, QEvent(QEvent.Enter))
    assert not grid._help_hide_timer.isActive()
    grid.eventFilter(grid._help_band, QEvent(QEvent.Leave))
    assert grid._help_hide_timer.isActive()
    assert grid.eventFilter(grid._table.viewport(), QEvent(QEvent.ToolTip))


def test_failed_help_render_keeps_last_readable_help(grid, monkeypatch):
    from spacr.qt.screens import settings_model
    before = grid._help.text()
    grid._help_pending = ""
    grid._show_pending_help()
    assert grid._help.text() == before
    def fail(*args):
        raise RuntimeError("help unavailable")
    monkeypatch.setattr(settings_model, "format_tooltip", fail)
    grid._help_pending = "cell_channel"
    grid._show_pending_help()
    assert grid._help.text() == before
    assert grid.settings()["cell_channel"] == 0


def test_help_api_opens_only_the_stored_link_and_survives_desktop_failure(grid, monkeypatch):
    calls = []
    monkeypatch.setattr(QtGui, "QDesktopServices",
                        SimpleNamespace(openUrl=lambda url: calls.append(url.toString())))
    grid._help_api_url = ""
    grid._help_api.clicked.emit()
    assert calls == []
    grid._help_api_url = "https://example.invalid/docs#cell_channel"
    grid._help_api.clicked.emit()
    assert calls == [grid._help_api_url]
    def fail(url):
        raise RuntimeError("desktop unavailable")
    monkeypatch.setattr(QtGui, "QDesktopServices", SimpleNamespace(openUrl=fail))
    grid._help_api.clicked.emit()
    assert grid.settings()["cell_channel"] == 0


def test_failed_animation_lookup_does_not_remove_written_help(grid, monkeypatch):
    from spacr import setting_animations
    def fail(key):
        raise RuntimeError("animation unavailable")
    monkeypatch.setattr(setting_animations, "animation_for_setting", fail)
    grid._write_help("Keep this explanation", key="cell_channel")
    assert grid._help.text() == "Keep this explanation"
    assert grid._help_offered_animation is None
    assert not grid._help_animation.isVisible()


def test_picker_failure_and_empty_click_leave_settings_unchanged(grid, monkeypatch):
    before = grid.settings()
    grid._cell_clicked(QModelIndex())
    def fail(obj):
        raise RuntimeError("picker unavailable")
    monkeypatch.setattr(grid, "choose_model_for", fail)
    source = grid._model.index(grid.questions().index(module.MODEL_QUESTION), 0)
    grid._table.clicked.emit(grid._table.model().mapFromSource(source))
    assert grid.settings() == before
    assert not grid.set_value("unknown", "cell", "2")
