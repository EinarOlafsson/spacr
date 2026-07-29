"""Alpha/Beta visibility is persisted and applied across the Qt interface."""

from __future__ import annotations

import pytest

from PySide6.QtCore import QSettings


@pytest.fixture
def maturity_prefs(tmp_path, monkeypatch):
    """Isolate every maturity test from the user's real QSettings."""
    from spacr.qt import preferences as prefs

    path = tmp_path / "maturity.ini"

    def _temporary_settings():
        return QSettings(str(path), QSettings.IniFormat)

    monkeypatch.setattr(prefs, "_settings", _temporary_settings)
    return prefs


def test_maturity_visibility_defaults_to_showing_everything(maturity_prefs):
    prefs = maturity_prefs
    assert prefs.get_show_alpha() is True
    assert prefs.get_show_beta() is True
    assert prefs.maturity_is_visible("alpha") is True
    assert prefs.maturity_is_visible("beta") is True
    assert prefs.maturity_is_visible("stable") is True


def test_maturity_visibility_round_trips_and_stable_cannot_be_hidden(
    maturity_prefs,
):
    prefs = maturity_prefs
    prefs.set_show_alpha(False)
    prefs.set_show_beta(False)

    assert prefs.get_show_alpha() is False
    assert prefs.get_show_beta() is False
    assert prefs.maturity_is_visible("alpha") is False
    assert prefs.maturity_is_visible("beta") is False
    assert prefs.maturity_is_visible("stable") is True
    assert prefs.maturity_is_visible("unknown-future-stage") is True


def test_preferences_dialog_loads_and_saves_both_feature_switches(
    qtbot,
    maturity_prefs,
    monkeypatch,
):
    from PySide6.QtWidgets import QCheckBox, QDialogButtonBox

    prefs = maturity_prefs
    monkeypatch.setattr(prefs, "apply_preferences_to_app", lambda *args: None)
    prefs.set_show_alpha(False)
    prefs.set_show_beta(True)

    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    alpha = dialog.findChild(QCheckBox, "ShowAlphaFeatures")
    beta = dialog.findChild(QCheckBox, "ShowBetaFeatures")

    assert alpha is not None and alpha.isChecked() is False
    assert beta is not None and beta.isChecked() is True

    alpha.setChecked(True)
    beta.setChecked(False)
    dialog.findChild(QDialogButtonBox).accepted.emit()

    assert prefs.get_show_alpha() is True
    assert prefs.get_show_beta() is False


def test_visible_app_registry_obeys_each_switch_independently(maturity_prefs):
    from spacr.qt.app import APPS, app_stage, visible_apps

    prefs = maturity_prefs
    prefs.set_show_alpha(False)
    visible = {row[0] for row in visible_apps()}
    assert all((app_stage(key) != "alpha") == (key in visible)
               for key, *_rest in APPS)

    prefs.set_show_alpha(True)
    prefs.set_show_beta(False)
    visible = {row[0] for row in visible_apps()}
    assert all((app_stage(key) != "beta") == (key in visible)
               for key, *_rest in APPS)

    prefs.set_show_alpha(False)
    visible = {row[0] for row in visible_apps()}
    assert visible
    assert all(app_stage(key) == "stable" for key in visible)


def test_home_contains_only_enabled_maturity_stages(
    qtbot,
    qt_theme_applied,
    maturity_prefs,
):
    from spacr.qt.app import make_home_page
    from spacr.qt.widgets.home import AppTile

    maturity_prefs.set_show_alpha(False)
    maturity_prefs.set_show_beta(True)
    page = make_home_page()
    qtbot.addWidget(page)

    stages = {tile.stage for tile in page.findChildren(AppTile)}
    assert "alpha" not in stages
    assert {"beta", "stable"} <= stages


def test_sidebar_refreshes_without_rebuilding_or_leaving_empty_headers(
    qtbot,
    qt_theme_applied,
    maturity_prefs,
):
    from PySide6.QtWidgets import QLabel
    from spacr.qt.app import Sidebar, app_stage

    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    buttons = {
        str(button.property("navKey")): button
        for button in sidebar._items
        if button.property("navKey") != "__home__"
    }

    maturity_prefs.set_show_alpha(False)
    maturity_prefs.set_show_beta(False)
    sidebar.refresh_visibility()

    for key, button in buttons.items():
        assert button.isHidden() == (app_stage(key) != "stable")
    headers = {
        label.text(): label
        for label in sidebar.findChildren(QLabel)
        if label.objectName() == "SidebarSection"
    }
    assert headers["Data"].isHidden(), "Data contains only Alpha modules"
    assert not headers["Core"].isHidden(), "Core retains stable modules"


def test_open_settings_toggle_in_place_without_losing_sections(
    qtbot,
    qt_theme_applied,
    maturity_prefs,
):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets.section import Section

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    sections = screen.findChildren(Section)
    beta = [section for section in sections if section.maturity() == "beta"]
    stable = [section for section in sections if section.maturity() == "stable"]
    assert beta and stable
    assert all(not section.isHidden() for section in sections)

    maturity_prefs.set_show_beta(False)
    screen.refresh_maturity_visibility()
    assert all(section.isHidden() for section in beta)
    assert all(not section.isHidden() for section in stable)
    assert not screen._maturity_notice.isHidden()
    assert "Beta settings are hidden" in screen._maturity_notice.text()

    maturity_prefs.set_show_beta(True)
    screen.refresh_maturity_visibility()
    assert all(not section.isHidden() for section in beta)
    assert screen._maturity_notice.isHidden()


def test_hidden_numeric_shortcut_does_not_open_a_filtered_module(
    maturity_prefs,
):
    from spacr.qt import shortcuts

    class Window:
        def __init__(self):
            self.opened = []

        def _on_nav_selected(self, key):
            self.opened.append(key)

    window = Window()
    maturity_prefs.set_show_beta(False)
    shortcuts._nav_by_index(window, 1)  # Timelapse is Beta.
    assert window.opened == []

    shortcuts._nav_by_index(window, 0)  # Mask is Stable.
    assert window.opened == ["mask"]


def test_command_palette_does_not_restore_hidden_modules(
    qtbot,
    maturity_prefs,
    monkeypatch,
):
    from PySide6.QtWidgets import QMainWindow
    from spacr import run_journal
    from spacr.qt.app import APPS, app_stage
    from spacr.qt.command_palette import CommandPalette

    maturity_prefs.set_show_alpha(False)
    maturity_prefs.set_show_beta(False)
    monkeypatch.setattr(run_journal, "recent_runs", lambda limit=8: [])
    window = QMainWindow()
    qtbot.addWidget(window)
    palette = CommandPalette(window)
    qtbot.addWidget(palette)

    app_labels = {
        command.label.removeprefix("Go to  ")
        for command in palette._commands
        if command.section.startswith("Apps ·")
    }
    expected = {
        name for key, name, _desc, _section in APPS
        if app_stage(key) == "stable"
    }
    assert app_labels == expected


def test_main_window_refreshes_home_dock_and_menus_together(
    qtbot,
    qt_theme_applied,
    maturity_prefs,
):
    from spacr.qt.app import MainWindow, app_stage
    from spacr.qt.widgets.home import AppTile

    maturity_prefs.set_show_alpha(False)
    maturity_prefs.set_show_beta(False)
    window = MainWindow()
    qtbot.addWidget(window)

    assert all(
        action.isVisible() == (app_stage(key) == "stable")
        for key, action in window._app_actions.items()
    )
    assert not window._demo_actions["timelapse"].isVisible()
    assert {
        tile.stage for tile in window._startup.findChildren(AppTile)
    } == {"stable"}

    maturity_prefs.set_show_alpha(True)
    window.refresh_theme()
    assert all(
        action.isVisible() == (app_stage(key) != "beta")
        for key, action in window._app_actions.items()
    )
    assert not window._demo_actions["timelapse"].isVisible()
