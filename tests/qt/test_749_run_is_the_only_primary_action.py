"""Run is the one primary action of a settings module (maintainer decision, 2026-11-01).

The UI audit found Run and Submit remote… both filled as primary buttons in
one row of about ten peers. Submit remote… is now a secondary (ghost) button,
Import settings… and Export settings… sit in the settings strip beside
Templates, and Clear console / Copy console sit in the console's header with the
Preferences gear kept right of Copy console, where it was asked for.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QPushButton

MODULES = ("mask", "measure", "classify")


@pytest.fixture(autouse=True)
def _private_recipes(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACR_RECIPE_DIR", str(tmp_path / "recipes"))


@pytest.fixture(params=MODULES)
def screen(request, qtbot):
    from spacr.qt import recipes, settings_search
    from spacr.qt.screens.app_screen import AppScreen

    page = AppScreen(request.param)
    qtbot.addWidget(page)
    assert settings_search.install(page) is not None
    assert recipes.install(page) is not None
    page.resize(1600, 1000)
    page.show()
    qtbot.waitExposed(page)
    return page


def test_run_is_the_only_primary_button_in_the_actions(screen):
    primary = [button for button in screen._actions_section.findChildren(QPushButton)
               if button.objectName() == "PrimaryButton"]
    assert primary == [screen._btn_run]
    assert screen._btn_remote.objectName() == "GhostButton"
    assert screen._actions_section.isAncestorOf(screen._btn_remote)


def test_import_and_export_sit_beside_templates(screen):
    bar = screen._settings_search
    for button in (screen._btn_import, screen._btn_export):
        assert bar.isAncestorOf(button) and button.isVisible()
        assert not screen._actions_section.isAncestorOf(button)
    controls = [widget for widget in bar._controls if not widget.isHidden()]
    at = controls.index(screen._recipe_button)
    assert controls[at - 2:at] == [screen._btn_import, screen._btn_export]
    for button in (screen._btn_import, screen._btn_export):
        assert not button.text() and not button.icon().isNull()
        assert button.width() <= 3 * button.height()
    assert screen._btn_import.accessibleName() == "Import settings…"
    assert screen._btn_export.accessibleName() == "Export settings…"


def test_console_actions_sit_in_the_console_header(qtbot, screen):
    from spacr.qt.widgets.activity_spinner import attach_activity_spinner

    card = screen._console_card
    title = card.title_label
    for button in (screen._btn_clear, screen._btn_copy_console):
        assert card.isAncestorOf(button) and button.isVisible()
        assert not screen._actions_section.isAncestorOf(button)
        assert not card.body.isAncestorOf(button)
        top = button.mapTo(card, button.rect().topLeft()).y()
        title_top = title.mapTo(card, title.rect().topLeft()).y()
        assert top < title_top + title.height() and title_top < top + button.height()
    gear = screen._btn_preferences
    assert gear.parentWidget() is screen._btn_copy_console.parentWidget()
    assert gear.x() > screen._btn_copy_console.x() > screen._btn_clear.x()
    assert attach_activity_spinner(screen) is None


def test_the_external_masks_source_column_keeps_its_caption(qtbot):
    """Its stretch column could shrink under its caption once the settings strip
    re-laid out; the column's floor is now its header's width."""
    from spacr.qt.widgets.external_mask_inputs import ExternalMaskInputWidget

    table = ExternalMaskInputWidget()
    qtbot.addWidget(table)
    table.resize(420, 300)
    table.show()
    qtbot.waitExposed(table)
    header = table._table.horizontalHeader()
    assert header.sectionSize(0) >= header.sectionSizeFromContents(0).width()
