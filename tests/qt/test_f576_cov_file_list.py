"""F576 store-folder picker and alpha refresh edges of the paired table."""
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog  # noqa: E402

from spacr.qt.widgets import file_list as fl  # noqa: E402

pytestmark = pytest.mark.qt


def test_store_folder_picker_cancel_adds_nothing(qtbot, monkeypatch):
    widget = fl.PairedFileTableWidget()
    qtbot.addWidget(widget)
    added = []
    monkeypatch.setattr(widget, "add_paths_for_side",
                        lambda paths, side: added.append(paths))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    before = widget.status.text()
    widget._pick_store_folder("score")
    assert added == []
    assert widget.status.text() == before


def test_store_folder_picker_refuses_a_plain_folder(qtbot, monkeypatch,
                                                    tmp_path):
    widget = fl.PairedFileTableWidget()
    qtbot.addWidget(widget)
    added = []
    monkeypatch.setattr(widget, "add_paths_for_side",
                        lambda paths, side: added.append((paths, side)))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path)))
    widget._pick_store_folder("score")
    assert added == []
    assert ".parquetdb" in widget.status.text()

    store = str(tmp_path / "m.parquetdb")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: store))
    widget._pick_store_folder("count")
    assert added == [([store], "count")]


def test_showing_the_table_refreshes_alpha_visibility(qtbot, monkeypatch):
    widget = fl.PairedFileTableWidget()
    qtbot.addWidget(widget)
    calls = []
    monkeypatch.setattr(widget, "_refresh_alpha_visibility",
                        lambda: calls.append(1))
    widget.show()
    qtbot.waitUntil(lambda: bool(calls))
    assert calls


def test_screen_refresh_without_paired_table(qtbot, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    from spacr.qt import preferences
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    screen = AppScreen("measure")
    try:
        assert "paired_data" not in screen._settings_model._widgets
        screen._refresh_alpha_visibility()
        model = screen._settings_model
        screen._settings_model = None
        try:
            screen._refresh_alpha_visibility()
        finally:
            screen._settings_model = model
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()
