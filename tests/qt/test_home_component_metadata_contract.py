"""Caller-owned Home metadata keeps tile actions and optional summaries distinct."""
from PySide6.QtCore import Qt
from PySide6.QtTest import QSignalSpy

from spacr.qt.widgets.home import AppTile, HomePage
from spacr.qt.widgets.module_hint_bar import ModuleHintBar
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


def test_supplied_home_metadata_without_a_summary_keeps_labels_actions_and_hint_ownership(
        qtbot, qt_theme_applied, private_ai_profile):
    preferences, _, _ = private_ai_profile
    preferences.set_ambient_enabled(False)
    preferences.set_reduce_motion(True)
    preferences.set_easy_mode(False)
    preferences.set_language('en')
    summary = 'Review a local image table.'
    apps = [('component-table', 'Image table', summary, 'Project views'),
            ('component-metadata', 'Metadata viewer', '', 'Project views')]
    supplied = list(apps)
    page = HomePage(apps, icon_provider=lambda key: None)
    qtbot.addWidget(page)
    actions = QSignalSpy(page.tile_clicked)
    page.resize(1200, 850)
    page.show()
    qtbot.waitUntil(page.isVisible, timeout=5000)
    assert not page.grab().isNull()
    tiles = {tile.property('moduleAppKey'): tile for tile in page.findChildren(AppTile)
             if tile.isVisibleTo(page)}
    assert set(tiles) == {'component-table', 'component-metadata'}
    assert tiles['component-table'].text_label == 'Image table'
    assert tiles['component-metadata'].text_label == 'Metadata viewer'
    assert tiles['component-metadata'].name_label.text() == 'Metadata viewer'
    hint = page.findChild(ModuleHintBar)
    assert hint is not None
    qtbot.mouseMove(page, page.rect().topLeft())
    qtbot.mouseMove(tiles['component-table'], tiles['component-table'].rect().center())
    qtbot.waitUntil(lambda: hint.module_key == 'component-table'
                    and hint.accessibleDescription() == summary, timeout=5000)
    assert page.show_module_hint('component-table') is True
    previous = hint.text()
    assert page.show_module_hint('component-metadata') is False
    assert hint.module_key == 'component-table' and hint.text() == previous
    assert hint.accessibleDescription() == summary
    page.resize(1040, 760)
    assert not page.grab().isNull()
    assert hint.module_key == 'component-table' and hint.accessibleDescription() == summary
    qtbot.mouseMove(tiles['component-metadata'], tiles['component-metadata'].rect().center())
    qtbot.waitUntil(lambda: hint.module_key == 'component-metadata', timeout=5000)
    assert summary not in hint.accessibleDescription()
    qtbot.mouseClick(tiles['component-metadata'], Qt.LeftButton)
    assert actions.count() == 1 and actions.at(0) == ['component-metadata']
    qtbot.mouseMove(tiles['component-table'], tiles['component-table'].rect().center())
    qtbot.waitUntil(lambda: hint.module_key == 'component-table'
                    and hint.accessibleDescription() == summary, timeout=5000)
    qtbot.mouseClick(tiles['component-table'], Qt.LeftButton)
    assert actions.count() == 2 and actions.at(1) == ['component-table']
    assert tiles['component-table'].name_label.text() == 'Image table'
    assert tiles['component-metadata'].name_label.text() == 'Metadata viewer'
    assert apps == supplied
