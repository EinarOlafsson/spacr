"""Help keeps documentation while native module routes remain in their menus."""
import pytest
from PySide6.QtWidgets import QMainWindow, QMenu

from spacr.qt import app
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


def _help_actions(window):
    menus = {menu.menuAction(): menu for menu in window.findChildren(QMenu)}
    pending, visited, actions = [window._easy_help_menu], set(), []
    while pending:
        menu = pending.pop()
        if menu in visited:
            continue
        visited.add(menu)
        for action in menu.actions():
            if action in menus:
                pending.append(menus[action])
            elif not action.isSeparator():
                actions.append(action)
    return actions


@pytest.mark.parametrize('navigator', [False, True])
def test_actual_help_has_no_duplicate_native_modules_and_keeps_help_routes(
        qtbot, private_ai_profile, monkeypatch, navigator):
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(navigator)
    preferences.set_reduce_motion(True)
    window = app.MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    try:
        actions = _help_actions(window)
        names = {app._easy_command_text(action.text().split('\t', 1)[0]) for action in actions}
        native_names = {app._easy_command_text(label) for action in window._app_actions.values()
                        for label in (action.text(), action.property('moduleNameSource')) if label}
        assert not names.intersection(native_names)
        assert {'navigator', 'keyboard shortcuts', 'tutorial', 'documentation',
                'about spacr', 'check for updates'} <= names
        for key, _label, _tip in app._HELP_MODULES:
            assert key in window._app_actions
        if 'feature_dict' not in window._app_actions:
            assert 'feature dictionary' in names
        called = []
        monkeypatch.setattr(window, '_on_nav_selected', called.append)
        monkeypatch.setattr(window, 'open_module', called.append)
        for key in ('db_browser', 'lineage'):
            action = window._app_actions[key]
            assert action.property('moduleAppKey') == key
            action.trigger()
        assert called == ['db_browser', 'lineage']
        assert app._easy_module_matches('Database Browser') == ('db_browser',)
        assert {'run_history', 'db_browser'} <= {row[0] for row in app.APPS}
    finally:
        window.close()


def test_late_nested_duplicates_are_removed_without_touching_help_only_tools(qtbot):
    window = QMainWindow()
    qtbot.addWidget(window)
    native = window.menuBar().addMenu('Modules')
    route = native.addAction('Measure')
    route.setProperty('moduleNameSource', 'Measure')
    route.setProperty('moduleAppKey', 'measure')
    route.setVisible(False)
    window._app_actions = {'measure': route}
    help_menu = window.menuBar().addMenu('Help')
    window._easy_help_menu = help_menu
    nested = QMenu('Advanced help', window)
    help_menu.addMenu(nested)
    duplicate = nested.addAction('&MEASURE…')
    translated = nested.addAction('Translated module label')
    translated.setProperty('moduleAppKey', 'measure')
    helper = nested.addAction('Run History')
    documentation = nested.addAction('Measurement documentation')
    opened = []
    route.triggered.connect(lambda: opened.append('measure'))
    documentation.triggered.connect(lambda: opened.append('documentation'))
    app._prune_native_help_actions(window)
    app._prune_native_help_actions(window)
    assert duplicate not in nested.actions() and translated not in nested.actions()
    assert helper in nested.actions() and documentation in nested.actions()
    assert route in native.actions()
    assert not route.isVisible()
    route.trigger()
    documentation.trigger()
    assert opened == ['measure', 'documentation']
