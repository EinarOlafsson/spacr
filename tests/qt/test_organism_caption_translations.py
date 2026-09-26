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
        payload = json.loads(review.read_text())
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
