"""Organism captions localize without changing assay routes or compartment keys."""
import json
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel


ROOT = Path(__file__).resolve().parents[2]
LANGUAGES = ('de', 'es', 'fr', 'sv', 'pt', 'is', 'zh_CN', 'ko', 'hi')


@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('app_key', ['toxoplasma', 'plasmodium', 'candida'])
def test_organism_labels_tooltips_and_compartment_keys(
        qtbot, qt_theme_applied, monkeypatch, language, app_key):
    from spacr.qt import i18n
    from spacr.qt.organisms import ORGANISMS
    from spacr.qt.screens.organism_screen import OrganismScreen

    targets = {}
    for review in sorted((ROOT / 'docs/i18n/reviewed/runtime' / language).glob('*.json')):
        payload = json.loads(review.read_text(encoding="utf-8"))
        for record in payload.get('records', []):
            if record.get('table') == 'ui' and not record.get('retired'):
                targets[record.get('source', record['key'])] = record['translation']
    monkeypatch.setattr(i18n, 'current_language', lambda: language)
    screen = OrganismScreen(app_key)
    qtbot.addWidget(screen)
    screen.resize(1100, 760)
    screen.show()
    qtbot.wait(10)
    assert screen._splitter.accessibleName() == targets['Information and modules divider']
    assert screen._splitter.handle(1).toolTip() == targets['Drag to resize the information pane.']
    headings = {label.text() for label in screen.findChildren(QLabel, 'OrganismSectionTitle')}
    assert {targets[title] for title, _, _ in ORGANISMS[app_key]['sections']} <= headings
    assert targets['Sources and research resources'] in headings
    from spacr.qt.organisms import workflow
    for tile, (key, title, description, icon) in zip(screen._tiles, ORGANISMS[app_key]['modules'], strict=True):
        live = bool(key or workflow(app_key, icon))
        assert tile.property('organismModuleKey') == (key or '')
        if title in targets:
            assert tile.text_label == tile.accessibleName() == targets[title]
            assert tile._name_lbl.full_text() == targets[title]
        assert targets[description] in tile.toolTip()
        assert tile.isEnabled() == live
        if not live:
            assert tile.toolTip().startswith(targets['Coming soon'] + ' — ')
            assert tile.accessibleDescription() == tile.toolTip()
    diagram = screen._diagram
    assert diagram.artwork.accessibleName() == targets['Cell compartments']
    assert diagram.clear_button.text() == targets['Clear components']
    assert diagram.clear_button.fontMetrics().horizontalAdvance(
        diagram.clear_button.text()) <= diagram.clear_button.contentsRect().width() - 20
    heading = 'hyperLOPIT compartment' if app_key == 'toxoplasma' else 'UniProt compartment'
    assert diagram.selector.accessibleName() == targets[heading]
    for index, (source, code) in enumerate(diagram.labels.items()):
        item = diagram.selector.item(index)
        assert item.data(Qt.UserRole) == code
        if source in targets:
            assert item.text() == targets[source]
        description = diagram.descriptions.get(code, '')
        if description in targets:
            assert item.toolTip() == targets[description]
            diagram._describe(code)
            assert targets[description] in diagram.caption.text()
    if app_key == 'toxoplasma':
        diagram._describe('SL0233')
        shared = ('These LOPIT classes share one anatomical outline; the diagram '
                  'does not distinguish their protein populations.')
        assert targets[shared] in diagram.caption.text()
        diagram._describe('SL0171')
        membrane = 'The membrane label uses the mitochondrial outline; it has no separate shape.'
        assert targets[membrane] in diagram.caption.text()
    diagram.selector.item(0).setCheckState(Qt.Checked)
    assert diagram.artwork.selected
    diagram.clear_button.click()
    assert not diagram.artwork.selected
    empty = ('Hover over the cell to identify a compartment. '
             'Check several labels to keep them highlighted.')
    assert diagram.caption.text() == targets[empty]


@pytest.mark.parametrize("scale", (0.75, 0.9, 1.25))
def test_hindi_clear_button_keeps_native_text_room_in_a_narrow_pane(
        qtbot, qt_theme_applied, monkeypatch, scale):
    from PySide6.QtWidgets import QApplication
    from spacr.qt import gui_scale, i18n, preferences
    from spacr.qt.screens.organism_screen import OrganismScreen

    settings = preferences._settings()
    keys = ("prefs/gui_scale", "prefs/font_scale")
    previous = {key: settings.value(key) for key in keys if settings.contains(key)}
    gui_scale.install_scaling_layer()
    with monkeypatch.context() as patch:
        patch.setattr(gui_scale, "_SCALE", scale)
        patch.setattr(i18n, "current_language", lambda: "hi")
        try:
            preferences.set_gui_scale(scale)
            preferences.set_font_scale(0.9)
            screen = OrganismScreen("toxoplasma")
            qtbot.addWidget(screen)
            screen.resize(800, 700)
            screen.show()
            qtbot.waitExposed(screen)
            diagram = screen._diagram
            button = diagram.clear_button
            hindi = button.text()
            assert hindi == "चयन साफ़ करें"
            button.setStyleSheet("font-size: 37px;")
            QApplication.processEvents()
            QApplication.processEvents()
            assert button.fontMetrics().horizontalAdvance(hindi) <= (
                button.contentsRect().width() - 20)
            assert diagram._legend.rect().contains(button.geometry())
            assert screen._scroll.horizontalScrollBar().maximum() == 0
            wide_floor = diagram._legend.minimumWidth()

            button.setText("Clear")
            QApplication.processEvents()
            QApplication.processEvents()
            assert diagram._legend.minimumWidth() < wide_floor
            text_floor = diagram._legend.minimumWidth()
            button.setStyleSheet("font-size: 12px;")
            QApplication.processEvents()
            QApplication.processEvents()
            assert diagram._legend.minimumWidth() < text_floor
            button.setText(hindi)
            QApplication.processEvents()
            QApplication.processEvents()
            assert button.text() == hindi
            assert button.fontMetrics().horizontalAdvance(hindi) <= (
                button.contentsRect().width() - 20)
            assert diagram._legend.rect().contains(button.geometry())
            assert screen._scroll.horizontalScrollBar().maximum() == 0
            screen.close()
        finally:
            for key in keys:
                if key in previous:
                    settings.setValue(key, previous[key])
                else:
                    settings.remove(key)


def test_native_wide_hindi_content_gets_pane_room_without_growing_the_window(
        qtbot, qt_theme_applied, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from spacr.qt import gui_scale, i18n, preferences
    from spacr.qt.screens.organism_screen import OrganismScreen

    settings = preferences._settings()
    keys = ("prefs/gui_scale", "prefs/font_scale")
    previous = {key: settings.value(key) for key in keys if settings.contains(key)}
    gui_scale.install_scaling_layer()
    with monkeypatch.context() as patch:
        patch.setattr(gui_scale, "_SCALE", 0.9)
        patch.setattr(i18n, "current_language", lambda: "hi")
        try:
            preferences.set_gui_scale(0.9)
            preferences.set_font_scale(0.9)
            screen = OrganismScreen("toxoplasma")
            qtbot.addWidget(screen)
            screen.resize(800, 700)
            screen.show()
            qtbot.waitExposed(screen)
            for _ in range(4):
                QApplication.processEvents()
            geometry = screen.geometry()
            button = screen._diagram.clear_button
            assert button.text() == "चयन साफ़ करें"
            button.setStyleSheet("font-size: 75px;")
            for _ in range(6):
                QApplication.processEvents()
            assert screen.geometry() == geometry
            assert button.fontMetrics().horizontalAdvance(button.text()) <= (
                button.contentsRect().width() - 20)
            assert screen._diagram._legend.rect().contains(button.geometry())
            assert screen._scroll.horizontalScrollBar().maximum() == 0
            assert screen._module_scroll.horizontalScrollBar().maximum() == 0
            wide_floor = screen._scroll.minimumWidth()
            button.setStyleSheet("font-size: 12px;")
            for _ in range(6):
                QApplication.processEvents()
            assert screen._scroll.minimumWidth() < wide_floor
            assert screen.geometry() == geometry
            assert screen._scroll.horizontalScrollBar().maximum() == 0
            assert screen._module_scroll.horizontalScrollBar().maximum() == 0
            screen.close()
        finally:
            for key in keys:
                if key in previous:
                    settings.setValue(key, previous[key])
                else:
                    settings.remove(key)
