"""Easy module visits use the real disclosure controls without saving overrides."""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFormLayout, QMainWindow, QStackedWidget

from spacr.qt import preferences
from spacr.qt.settings_search import (
    ALL, ESSENTIALS, SettingsSearchBar, disclosure_for, forget_disclosure,
    install_window_hooks, remember_disclosure,
)


@pytest.fixture
def disclosure_state():
    previous = preferences.get_easy_mode()
    forget_disclosure()
    yield
    preferences.set_easy_mode(previous)
    forget_disclosure()


@pytest.mark.parametrize("app_key", ["mask", "measure"])
def test_easy_modules_start_essential_and_allow_native_expansion(qtbot, disclosure_state, app_key):
    from spacr.qt.screens.app_screen import AppScreen

    remember_disclosure(app_key, ALL)
    preferences.set_easy_mode(True)
    window = QMainWindow()
    qtbot.addWidget(window)
    stack = window._stack = QStackedWidget(window)
    window.setCentralWidget(stack)
    screen = AppScreen(app_key)
    stack.addWidget(screen)
    watcher = install_window_hooks(window)
    window.resize(1400, 1000)
    window.show()
    qtbot.waitUntil(lambda: getattr(screen, "_settings_search", None) is not None)
    bar = screen._settings_search
    assert watcher is not None and bar.level() == ESSENTIALS
    screen.refresh_maturity_visibility()
    assert not screen.dimension_is_on("t")
    if app_key == "measure":
        assert not {"timelapse", "timelapse_objects"} & set(bar.visible_keys())
    assert 0 < len(bar.visible_keys()) < len(bar.indexed_keys())
    essential = set(bar.visible_keys())
    hidden_rows = [(section, field) for key, (section, field) in bar._index.items()
                   if key not in essential and field is not None
                   and section.findChild(QFormLayout) is not None]
    assert hidden_rows
    assert any(not section.findChild(QFormLayout).isRowVisible(field)
               for section, field in hidden_rows)
    qtbot.mouseClick(bar._disclosure, Qt.LeftButton)
    assert bar.level() == ALL
    assert set(bar.visible_keys()) > essential
    assert any(section.findChild(QFormLayout).isRowVisible(field)
               for section, field in hidden_rows)
    bar.apply()
    screen._settings_model.refresh_object_visibility()
    assert bar.level() == ALL
    qtbot.mouseClick(bar._disclosure, Qt.LeftButton)
    assert bar.level() == ESSENTIALS
    assert disclosure_for(app_key) == ALL
    window.close()


def test_cached_visits_reset_easy_disclosure_and_restore_normal_choice(qtbot, disclosure_state, tmp_path):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.screens.foreign import ForeignScreen

    remember_disclosure("mask", ALL)
    preferences.set_easy_mode(True)
    window = QMainWindow()
    qtbot.addWidget(window)
    stack = window._stack = QStackedWidget(window)
    window.setCentralWidget(stack)
    screen = AppScreen("mask")
    native = ForeignScreen()
    stack.addWidget(screen)
    stack.addWidget(native)
    watcher = install_window_hooks(window)
    window.resize(1400, 1000)
    window.show()
    qtbot.waitUntil(lambda: getattr(screen, "_settings_search", None) is not None)
    bar = screen._settings_search
    qtbot.mouseClick(bar._disclosure, Qt.LeftButton)
    assert bar.level() == ALL
    stack.setCurrentWidget(native)
    assert watcher.install_current() is None
    assert not native.findChildren(SettingsSearchBar)
    assert not hasattr(native, "_settings_search")
    assert native._images_edit.isVisible() and native._images_edit.isEnabled()
    qtbot.keyClicks(native._images_edit, str(tmp_path))
    assert native._images_edit.text() == str(tmp_path)
    stack.setCurrentWidget(screen)
    assert screen._settings_search is bar and bar.level() == ESSENTIALS
    assert disclosure_for("mask") == ALL
    preferences.set_easy_mode(False)
    stack.setCurrentWidget(native)
    assert native._images_edit.text() == str(tmp_path)
    stack.setCurrentWidget(screen)
    assert bar.level() == ALL and disclosure_for("mask") == ALL
    qtbot.mouseClick(bar._disclosure, Qt.LeftButton)
    assert bar.level() == ESSENTIALS and disclosure_for("mask") == ESSENTIALS
    stack.setCurrentWidget(native)
    stack.setCurrentWidget(screen)
    assert bar.level() == ESSENTIALS
    bar._show_all_without_remembering()
    bar.apply()
    stack.setCurrentWidget(native)
    stack.setCurrentWidget(screen)
    assert bar.level() == ALL and disclosure_for("mask") == ESSENTIALS
    window.close()
