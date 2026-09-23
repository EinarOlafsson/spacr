"""The application dropdown uses the organism page's own module catalogue."""
import pytest

from spacr.qt import app
from spacr.qt.organisms import ORGANISMS


@pytest.fixture
def window(qtbot, qt_theme_applied):
    window = app.MainWindow()
    qtbot.addWidget(window)
    return window


def test_assays_menu_groups_children_under_the_three_organisms(window):
    menu = window._section_menus[app.SECTION_ASSAYS]
    keys = [action.property('moduleAppKey') or
            (action.menu().property('moduleAppKey') if action.menu() else None)
            for action in menu.actions()]
    assert keys[:3] == list(ORGANISMS)
    assert 'analyze_plaques' not in keys and 'host_pathogen' not in keys
    for key, guide in ORGANISMS.items():
        actions = [a for a in window._organism_menus[key].actions() if not a.isSeparator()]
        assert actions[0].property('moduleAppKey') == key
        assert len(actions) == len(guide['modules']) + 1
        for action, (route, title, description, icon) in zip(actions[1:], guide['modules']):
            assert title in action.text()
            if route:
                assert action.property('moduleAppKey') == route
            else:
                assert not action.isEnabled() and 'Coming soon' in action.text()


def test_assay_actions_open_their_modules_and_starplast_uses_its_installer(window, monkeypatch):
    opened = []
    monkeypatch.setattr(window, 'open_module', opened.append)
    monkeypatch.setattr('spacr.qt.starplast.open_starplast', lambda parent: opened.append('starplast'))
    for route, *_ in ORGANISMS['toxoplasma']['modules']:
        if route:
            window._app_actions[route].trigger()
    assert opened == [entry[0] for entry in ORGANISMS['toxoplasma']['modules'] if entry[0]]


def test_hiding_an_organism_hides_its_menu_and_preserves_its_overview_action(window, monkeypatch):
    monkeypatch.setattr(app, 'app_is_visible', lambda key: key != 'plasmodium')
    window._refresh_app_action_visibility()
    assert not window._organism_menus['plasmodium'].menuAction().isVisible()
    selected = []
    monkeypatch.setattr(window, '_on_nav_selected', selected.append)
    window._app_actions['toxoplasma'].trigger()
    assert selected == ['toxoplasma']
