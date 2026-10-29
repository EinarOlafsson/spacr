"""Reviewed Navigator task phrases produce native suggestion cards on Enter."""
import re
from pathlib import Path

import pytest
from PySide6.QtCore import Qt

from spacr.qt import app
from spacr.qt.widgets.home import _EasyHomePage
from tests.qt.test_733_easy_commands import commands, overview_seed_route
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


_ITEM = Path(__file__).resolve().parents[2] / 'features/feature_list/733_easy_mode_content_guided_home.txt'
_PAIRS = [(original, alternate, tuple(keys.split(', ')))
          for original, alternate, keys in re.findall(
              r'^\d{3}\. (.+) \| (.+) -> (.+)$', _ITEM.read_text(), re.M)]
_CASES = [(original, keys) for original, _alternate, keys in _PAIRS]
_ALTERNATES = [(alternate, keys) for _original, alternate, keys in _PAIRS]
assert len(_CASES) == len(_ALTERNATES) == 100


@pytest.mark.parametrize('phrase,expected', _CASES)
def test_reviewed_tasks_offer_cards_without_opening_modules(commands, qtbot, monkeypatch, phrase, expected):
    monkeypatch.setattr(app, 'app_is_visible', lambda key: True)
    page = _EasyHomePage(commands)
    commands.setCentralWidget(page)
    commands._easy_navigation_suggestions = lambda value: app.MainWindow._easy_navigation_suggestions(commands, value)
    page.command_requested.connect(commands._open_easy_command)
    emitted = []
    page.command_requested.connect(emitted.append)
    query = 'Can you help me ' + phrase.upper() + '?'
    rows = commands._easy_navigation_suggestions(query)
    assert tuple(row[0] for row in rows) == expected
    assert len(rows) == len({row[0] for row in rows})
    assert all(row[1] and row[2] and not row[3] for row in rows)
    page.path_edit.setText(query)
    assert not page.field.cards
    assert not commands.opened
    page._submit()
    assert tuple(card._key for card in page.field.cards) == expected
    assert not commands.opened and not commands.navigated and not emitted
    assert all(not row[0].startswith('__easy_command__:') for row in rows)


@pytest.mark.parametrize('value', [
    '/actual/image analysis/image.tif', r'C:\images\object detection.tif',
    './mask curation', '~/make a graph', 'read a database.sqlite',
    'image analysis.tif', 'analyze spatial imaging.npy',
])
def test_task_phrases_never_override_explicit_file_paths(value):
    assert not app._easy_task_keys(value)
    assert not app._easy_module_search(value)


def test_task_cards_keep_maturity_filter_and_existing_module_partial_search(commands, qtbot, monkeypatch):
    page = _EasyHomePage(commands)
    commands.setCentralWidget(page)
    commands._easy_navigation_suggestions = lambda value: app.MainWindow._easy_navigation_suggestions(commands, value)
    monkeypatch.setattr(app, 'app_is_visible', lambda key: key == 'make_masks')
    page.path_edit.setText('OBJECT DETECTION!')
    page._submit()
    assert [card._key for card in page.field.cards] == ['make_masks']
    assert not commands.opened
    assert 'measure' in {row[0] for row in app._easy_module_search('MeAsUr')}
    assert 'db_browser' in {row[0] for row in app._easy_module_search('DaTaBaSe')}
    assert page.path_edit.placeholderText() == 'Drop a file or folder | Type a Path | Ask a Question ◉'


@pytest.mark.parametrize('phrase,expected', [
    ('image data', ('mask', 'measure', 'qc_dashboard')),
    ('make figure', ('graph_builder',)),
    ('MAKE GRAPH!', ('graph_builder',)),
    ('high content image', ('mask', 'measure', 'qc_dashboard')),
])
def test_short_forms_and_partial_tasks_use_native_cards(commands, phrase, expected):
    rows = app.MainWindow._easy_navigation_suggestions(commands, phrase)
    assert tuple(row[0] for row in rows) == expected
    assert not commands.opened


def test_existing_relative_task_named_folder_keeps_file_inspection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'image data').mkdir()
    assert app._easy_task_keys('image data') == ()
    assert app._easy_module_search('image data') == []


def test_spatial_task_keeps_declared_page_fold_maturity(commands, monkeypatch):
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, 'maturity_is_visible', lambda stage: stage != 'alpha')
    rows = app.MainWindow._easy_navigation_suggestions(commands, 'analyze spatial imaging')
    assert rows == []
    assert not commands.opened


@pytest.mark.parametrize('phrase', ['image analasys', 'HIGH CONTENT IMAGE ANALASYS'])
def test_user_supplied_analysis_spellings_offer_image_tools(commands, phrase):
    rows = app.MainWindow._easy_navigation_suggestions(commands, phrase)
    assert tuple(row[0] for row in rows) == ('mask', 'measure', 'qc_dashboard')
    assert not commands.opened and not commands.navigated


@pytest.mark.parametrize('phrase,expected', _ALTERNATES)
def test_paired_alternate_tasks_offer_cards_without_opening_modules(
        commands, qtbot, monkeypatch, phrase, expected):
    test_reviewed_tasks_offer_cards_without_opening_modules(
        commands, qtbot, monkeypatch, phrase, expected)


@pytest.mark.parametrize('phrase', [
    'analyze masks', 'ANALYSE MASKS', 'mask analysis',
    'analyze segmentation masks', 'segmentation mask analysis',
])
def test_existing_mask_analysis_tasks_offer_native_cards_on_enter(
        commands, qtbot, monkeypatch, phrase):
    test_reviewed_tasks_offer_cards_without_opening_modules(
        commands, qtbot, monkeypatch, phrase, ('measure', 'qc_dashboard', 'make_masks'))


@pytest.mark.parametrize('phrase', [
    'detect features', 'FEATURE DETECTION', 'detect image features',
    'detect morphological features', 'extract features', 'feature extraction',
])
def test_feature_detection_tasks_offer_measure_without_launching(
        commands, qtbot, monkeypatch, phrase):
    test_reviewed_tasks_offer_cards_without_opening_modules(
        commands, qtbot, monkeypatch, phrase, ('measure',))

def test_hidden_native_route_keeps_the_current_page_and_reports_the_preference(
        qtbot, private_ai_profile):
    from spacr.qt import preferences

    preferences.set_show_alpha(False)
    window = app.MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    hidden = next(key for key, *_ in app.APPS if not app.app_is_visible(key))
    before = window._stack.currentWidget()
    existing = dict(window._screens)
    try:
        assert window._open_easy_named_module(hidden) is None
        assert window._stack.currentWidget() is before and window._screens == existing
        assert 'hidden by the current application preferences' in window.statusBar().currentMessage()
    finally:
        window.close()


def test_canceling_actual_saved_settings_preview_does_not_open_or_apply_the_module(
        qtbot, private_ai_profile, monkeypatch, tmp_path):
    import json
    from copy import deepcopy
    from PySide6.QtWidgets import QDialog, QDialogButtonBox, QPlainTextEdit

    window = app.MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    source = tmp_path / 'settings.csv'
    source.write_text('setting,value\nsrc,unchanged\n')
    before_bytes = source.read_bytes()
    settings = {'src': str(tmp_path), 'number_of_organelles': 1}
    context = {'saved_settings': {'module': 'measure', 'path': str(source), 'settings': settings}}
    original = deepcopy(context)
    before = window._stack.currentWidget()
    screens = dict(window._screens)
    shown = []

    def cancel(dialog):
        dialog.show()
        qtbot.waitExposed(dialog)
        preview = dialog.findChild(QPlainTextEdit, 'EasySettingsPreview')
        shown.append(json.loads(preview.toPlainText()))
        assert preview.isReadOnly()
        button = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Cancel)
        qtbot.mouseClick(button, Qt.LeftButton)
        assert dialog.result() == QDialog.Rejected
        return dialog.result()

    monkeypatch.setattr(QDialog, 'exec', cancel)
    try:
        assert window._open_easy_input('measure', str(source), context)
        assert shown == [settings]
        assert window._stack.currentWidget() is before and window._screens == screens
        assert context == original and source.read_bytes() == before_bytes
    finally:
        window.close()


@pytest.mark.parametrize('inferred', [False, True])
@pytest.mark.parametrize('destination', ['accept', 'cancel', 'unavailable', 'cannot-apply'])
def test_settings_suggestion_preserves_source_and_applies_only_after_acceptance(
        commands, monkeypatch, tmp_path, inferred, destination):
    from copy import deepcopy

    path = str(tmp_path / 'settings.csv')
    settings = {'src': str(tmp_path), 'number_of_organelles': 2,
                'organelle_mask_dim': 0, 'organelleb_mask_dim': 1}
    source = deepcopy(settings)
    context = ({'submodule_settings': {'measure': {'settings': settings}}}
               if inferred else {'saved_settings': {'module': 'measure',
                                                    'path': path, 'settings': settings}})
    previews, applied = [], []

    def preview(module, values, **options):
        previews.append((module, dict(values), options))
        return destination != 'cancel'

    monkeypatch.setattr(commands, '_preview_easy_settings', preview)
    if destination == 'unavailable':
        monkeypatch.setattr(commands, '_open_easy_named_module', lambda key: None)
    elif destination != 'cannot-apply':
        commands.screen.apply_settings_dict = applied.append
    assert commands._open_easy_input('measure', path, context)
    assert previews == [('measure', source, {'inferred': True} if inferred else {})]
    assert settings == source
    if destination == 'accept':
        assert applied == [source] and applied[0] is not settings
    else:
        assert applied == []
    if destination == 'cancel':
        assert commands.opened == []
    if destination == 'cannot-apply':
        assert 'cannot apply' in commands.statusBar().currentMessage()


@pytest.mark.parametrize('accepted', [False, True])
def test_ambiguous_real_help_actions_require_an_explicit_choice(commands, monkeypatch, accepted):
    from PySide6.QtWidgets import QInputDialog

    triggered = []
    for name in ('First', 'Second'):
        menu = commands._easy_help_menu.addMenu(name)
        menu.addAction('Inspect result').triggered.connect(
            lambda checked=False, name=name: triggered.append(name))
    disabled = commands._easy_help_menu.addAction('Inspect result')
    disabled.setEnabled(False)
    choices = []

    def choose(parent, title, label, items, *args):
        choices.append(list(items))
        return items[0], accepted

    monkeypatch.setattr(QInputDialog, 'getItem', choose)
    commands._open_easy_command('inspect result')
    assert choices == [['Inspect result', 'Inspect result']]
    assert triggered == (['Second'] if accepted else [])
    assert not commands.opened and not commands.urls


@pytest.mark.parametrize('route', ['make_masks', 'db_browser', 'foreign', 'convert'])
def test_native_suggestion_delivers_the_exact_path_to_its_existing_screen_adapter(
        commands, qtbot, monkeypatch, tmp_path, route):
    import sqlite3
    from types import SimpleNamespace
    import numpy as np
    import tifffile
    from spacr.qt.screens.convert import ConvertScreen
    from spacr.qt.screens.db_browser import DbBrowserScreen
    from spacr.qt.screens.foreign import ForeignScreen
    from spacr.qt.screens.make_masks import MakeMasksScreen
    from tests.test_scn_reader import write_scn

    pixels = np.arange(8 * 9, dtype=np.uint16).reshape(8, 9)
    if route == 'db_browser':
        source = tmp_path / 'measurements.db'
        with sqlite3.connect(source) as connection:
            connection.execute('CREATE TABLE cells (area REAL)')
            connection.execute('INSERT INTO cells VALUES (12)')
        screen = DbBrowserScreen(threaded=False)
    elif route == 'foreign':
        source = tmp_path / 'source.csv'
        source.write_text('area,intensity\n12,34\n')
        screen = ForeignScreen(threaded=False)
    elif route == 'convert':
        source = tmp_path / 'source.scn'
        write_scn(source, [pixels])
        screen = ConvertScreen(threaded=False)
    else:
        source = tmp_path / 'source.tif'
        tifffile.imwrite(source, pixels)
        screen = MakeMasksScreen()
    path = source.parent if route == 'convert' else source
    original = source.read_bytes()
    qtbot.addWidget(screen)
    commands.screen = screen
    received = []
    names = {'make_masks': 'open_paths', 'db_browser': 'set_database',
             'foreign': 'set_measurements', 'convert': 'set_source'}
    native = getattr(screen, names[route])
    def receive(value):
        received.append(value)
        return native(value)
    monkeypatch.setattr(screen, names[route], receive)
    commands._startup = SimpleNamespace(_last_result={})
    try:
        app.MainWindow._open_easy_suggestion(commands, route, str(path))
        assert received == ([[str(path)]] if route == 'make_masks' else [str(path)])
        assert commands.opened == [route]
        if route == 'make_masks':
            qtbot.waitUntil(lambda: not screen._loading and screen._reader_worker is None
                            and screen._load_worker is None, timeout=20000)
            np.testing.assert_array_equal(screen._canvas.image, pixels)
        elif route == 'db_browser':
            assert screen.tables() == ['cells'] and screen._db.path == str(source)
        elif route == 'foreign':
            assert screen.measurements_path() == str(source)
        else:
            assert screen.source_path() == str(source.parent)
        assert source.read_bytes() == original
    finally:
        screen.close()


@pytest.mark.parametrize('apply_available', [False, True])
def test_inspected_settings_overrides_keep_the_source_and_use_native_apply_when_available(
        commands, monkeypatch, tmp_path, apply_available, overview_seed_route):
    from types import SimpleNamespace

    folder = str(tmp_path)
    overrides = {overview_seed_route: {'diameter': 19, 'channels': [0]}}
    commands._startup = SimpleNamespace(_last_result={}, _settings_overrides=overrides)
    received, trained = [], []
    if apply_available:
        commands.screen.apply_settings_dict = received.append
    monkeypatch.setattr(commands, 'open_module', lambda key: commands.screen, raising=False)
    monkeypatch.setattr(commands, '_on_train_requested', lambda key, values: trained.append((key, values)), raising=False)
    monkeypatch.setattr(app, '_opened_module_screen', lambda *args: commands.screen)
    monkeypatch.setattr(app, 'app_is_visible', lambda key: True)
    app.MainWindow._open_easy_suggestion(commands, overview_seed_route, folder)
    expected = {'src': folder, 'diameter': 19, 'channels': [0]}
    assert received == ([expected] if apply_available else [])
    assert trained == ([] if apply_available else [(overview_seed_route, expected)])
    assert overrides == {overview_seed_route: {'diameter': 19, 'channels': [0]}}


@pytest.mark.parametrize('failure', ['refused', 'io-error', 'value-error', 'accepted'])
def test_analysis_input_handler_refusal_keeps_the_screen_and_source_intact(
        commands, monkeypatch, tmp_path, failure):
    from spacr.qt import dnd_handlers

    path = tmp_path / 'measurements.db'
    path.write_bytes(b'existing database')
    called = []

    class Handler:
        def accepts_multiple(self):
            return False

        def can_accept(self, value):
            assert value == path
            return failure != 'refused'

        def error_message(self, value):
            return 'Unsupported measurement schema'

        def apply(self, value, screen):
            called.append((value, screen))
            if failure == 'io-error':
                raise OSError('The database is no longer readable')
            if failure == 'value-error':
                raise ValueError('The selected table has no object identity')

    monkeypatch.setattr(dnd_handlers, 'get_handler', lambda key: Handler())
    assert commands._open_easy_input('graph_builder', str(path),
                                     {'analysis_inputs': {'graph_builder': {}}})
    assert called == ([] if failure == 'refused' else [(path, commands.screen)])
    assert commands.opened == ['graph_builder']
    assert path.read_bytes() == b'existing database'
    if failure != 'accepted':
        assert commands.statusBar().currentMessage()


@pytest.mark.parametrize('accepted', [False, True])
def test_multi_plate_choice_is_explicit_and_deduplicates_native_database_inputs(
        commands, monkeypatch, tmp_path, accepted):
    from PySide6.QtWidgets import QInputDialog

    first, second = str(tmp_path / 'first.db'), str(tmp_path / 'second.db')
    context = {'plate_inputs': [{'database': first}, {'database': second}, {'database': first}],
               'analysis_inputs': {'graph_builder': {}}}
    offers = []

    class Handler:
        def accepts_multiple(self):
            return True

    def choose(parent, title, label, items, *args):
        offers.append(items)
        return items[0], accepted

    monkeypatch.setattr(QInputDialog, 'getItem', choose)
    result = commands._easy_input_paths('graph_builder', first, context, Handler())
    assert offers == [['All compatible plates', first, second]]
    assert result == ([first, second] if accepted else [])
    assert len(context['plate_inputs']) == 3


@pytest.mark.parametrize('key,dimension', [('3d', 'z'), ('timelapse', 't')])
def test_dimension_tasks_enable_the_existing_mask_dimension_without_another_screen(
        commands, monkeypatch, key, dimension):
    requests, dimensions = [], []
    commands._screens = {'mask': commands.screen}
    commands.screen.set_dimension = lambda name, enabled: dimensions.append((name, enabled))
    monkeypatch.setattr(commands, 'open_module', requests.append, raising=False)
    monkeypatch.setattr(app, 'app_is_visible', lambda name: True)
    result = app.MainWindow._open_easy_named_module(commands, key)
    assert result is commands.screen
    assert requests == ['mask'] and dimensions == [(dimension, True)]
    assert list(commands._screens) == ['mask']


@pytest.mark.parametrize('installation', ['method', 'mapping', 'sequence'])
def test_folded_native_task_uses_its_existing_owned_page_opener(commands, monkeypatch, installation):
    from types import SimpleNamespace
    from PySide6.QtWidgets import QWidget

    page = QWidget(commands.screen)
    opened = []

    def open_page(*args):
        opened.append(args)
        return page

    if installation == 'method':
        commands.screen.open_folded = open_page
    else:
        opener = SimpleNamespace(key='ops', open=open_page)
        commands.screen._fold_openers = ({'ops': opener} if installation == 'mapping'
                                          else [opener])
    commands._screens = {'mask': commands.screen}
    monkeypatch.setattr(commands, 'open_module', lambda key: None, raising=False)
    monkeypatch.setattr(app, 'app_is_visible', lambda key: True)
    result = app.MainWindow._open_easy_named_module(commands, 'ops')
    assert result is page
    assert opened == ([('ops',)] if installation == 'method' else [()])
    assert commands._screens == {'mask': commands.screen}


@pytest.mark.parametrize('text,kind', [('', 'inspect'), ('unrecognized task qqq', 'inspect'),
                                     ('navigator', 'easy'), ('preferences', 'preferences'),
                                     ('pipeline overview', 'sample'), ('github', 'url'),
                                     ('api', 'url'), ('about', 'menu')])
def test_explicit_non_module_commands_dispatch_only_the_existing_destination(
        commands, monkeypatch, text, kind):
    called = []
    commands._easy_help_menu.addAction('About spaCR').triggered.connect(
        lambda: called.append('menu'))
    commands._act_preferences.triggered.connect(lambda: called.append('preferences'))
    monkeypatch.setattr(commands, '_set_easy_mode_from_help', lambda enabled: called.append('easy'))
    monkeypatch.setattr(commands, '_start_a_sample_project', lambda: called.append('sample'), raising=False)
    commands._open_easy_command(text)
    if kind == 'inspect':
        assert commands.inspections == [True] and not called and not commands.urls
    elif kind == 'url':
        assert len(commands.urls) == 1 and commands.urls[0].startswith('https://')
        assert not called and not commands.inspections
    else:
        assert called == [kind] and not commands.urls and not commands.inspections
    assert not commands.opened


@pytest.mark.parametrize('visible', [False, True])
def test_overview_suggestion_binds_the_real_owner_source_and_respects_hidden_preferences(
        commands, qtbot, monkeypatch, tmp_path, visible):
    from types import SimpleNamespace
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets import qc_field_browser

    source = tmp_path / 'field.tif'
    source.write_bytes(b'existing field')
    owner = AppScreen('mask')
    qtbot.addWidget(owner)
    prior = owner._settings_model.collect()
    commands._startup = SimpleNamespace(_last_result={}, _workflow=SimpleNamespace(
        data={'modules': {'overview': {'route': 'overview', 'host_key': 'mask'}}}))
    opened, requests = [], []
    monkeypatch.setattr(commands, 'open_module', lambda key: opened.append(key) or owner, raising=False)
    monkeypatch.setattr(app, '_opened_module_screen', lambda *args: owner)
    monkeypatch.setattr(app, 'app_is_visible', lambda key: visible)
    monkeypatch.setattr(qc_field_browser, '_open_overview_module',
                        lambda screen, settings: requests.append((screen, settings)))
    app.MainWindow._open_easy_suggestion(commands, 'overview', str(source))
    if visible:
        assert opened == ['mask'] and len(requests) == 1
        assert requests[0][0] is owner and requests[0][1]['src'] == str(tmp_path)
    else:
        assert not opened and not requests
        assert 'hidden' in commands.statusBar().currentMessage()
    assert owner._settings_model.collect() == prior and source.read_bytes() == b'existing field'
    owner.close()


@pytest.mark.parametrize('accepts', [False, True])
def test_workflow_native_route_delivers_only_schema_accepted_input(
        commands, monkeypatch, tmp_path, accepts, overview_seed_route):
    from types import SimpleNamespace
    from spacr.qt import dnd_handlers

    source = tmp_path / 'source.db'
    source.write_bytes(b'original input')
    commands._startup = SimpleNamespace(_last_result={}, _workflow=SimpleNamespace(
        data={'modules': {}}, native_routes={overview_seed_route}))
    delivered = []

    class Handler:
        def can_accept(self, path):
            assert path == source
            return accepts

        def apply(self, path, screen):
            delivered.append((path, screen))

    monkeypatch.setattr(commands, 'open_module', lambda key: commands.screen, raising=False)
    monkeypatch.setattr(app, '_opened_module_screen', lambda *args: commands.screen)
    monkeypatch.setattr(app, 'app_is_visible', lambda key: True)
    monkeypatch.setattr(dnd_handlers, 'get_handler', lambda key: Handler())
    app.MainWindow._open_easy_suggestion(commands, overview_seed_route, str(source))
    assert delivered == ([(source, commands.screen)] if accepts else [])
    assert source.read_bytes() == b'original input'


@pytest.mark.parametrize('at_home', [False, True])
def test_disabling_easy_mode_retains_an_active_module_and_only_home_stays_home(
        commands, monkeypatch, at_home):
    from PySide6.QtWidgets import QStackedWidget, QWidget
    from spacr.qt import preferences

    stack = QStackedWidget(commands)
    home = QWidget(stack)
    stack.addWidget(home)
    stack.addWidget(commands.screen)
    stack.setCurrentWidget(home if at_home else commands.screen)
    commands.setCentralWidget(stack)
    commands._startup, commands._stack = home, stack
    preferences.set_easy_mode(True)
    app.MainWindow._set_easy_mode_from_help(commands, False)
    assert not preferences.get_easy_mode() and not commands._act_easy_mode.isChecked()
    assert commands.refreshed == [{'easy_mode'}]
    assert commands.navigated == (['__home__'] if at_home else [])
    assert stack.currentWidget() is (home if at_home else commands.screen)


def test_navigation_search_offers_only_visible_enabled_help_actions_without_duplicates(commands):
    menu = commands._easy_help_menu.addMenu('Audit')
    menu.addAction('Export audit report')
    menu.addAction('Export audit report')
    menu.addSeparator()
    menu.addAction('Hidden audit report').setVisible(False)
    menu.addAction('Disabled audit report').setEnabled(False)
    rows = app.MainWindow._easy_navigation_suggestions(commands, 'audit report')
    assert [(key, title) for key, title, description, path in rows] == [
        ('__easy_command__:Export audit report', 'Export audit report')]
    assert not commands.opened and not commands.navigated


def test_native_help_pruning_handles_a_shared_real_submenu_without_removing_documentation(commands):
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QMenu

    untouched = commands._easy_help_menu.addAction('API documentation')
    app._prune_native_help_actions(commands)
    assert untouched in commands._easy_help_menu.actions()
    native = QAction('Mask', commands)
    commands._app_actions = {'mask': native}
    branch = commands._easy_help_menu.addMenu('More help')
    shared = QMenu('Shared module help', commands)
    commands._easy_help_menu.addMenu(shared)
    branch.addMenu(shared)
    duplicate = shared.addAction('Open this module')
    duplicate.setProperty('moduleAppKey', 'mask')
    genuine = shared.addAction('Read module documentation')
    app._prune_native_help_actions(commands)
    assert duplicate not in shared.actions()
    assert genuine in shared.actions() and untouched in commands._easy_help_menu.actions()
    assert shared.menuAction() in branch.actions()
    assert shared.menuAction() in commands._easy_help_menu.actions()


@pytest.mark.parametrize('kind', ['dimension', 'paired', 'analysis'])
def test_input_routing_with_an_unavailable_destination_cannot_apply_or_change_the_source(
        commands, monkeypatch, tmp_path, kind):
    source = tmp_path / 'source.tif'
    source.write_bytes(b'unchanged input')
    requests = []

    def unavailable(key):
        requests.append(key)
        return None

    monkeypatch.setattr(commands, '_open_easy_named_module', unavailable)
    if kind == 'dimension':
        key, context = '3d', {'dimension_routes': {'3d': {'source': str(source)}}}
    elif kind == 'paired':
        key, context = 'make_masks', {'paired_layout': {'kind': 'unstructured', 'pairs': {str(source): 'mask.tif'}}}
    else:
        key, context = 'graph_builder', {'analysis_inputs': {'graph_builder': {}}}
    assert commands._open_easy_input(key, str(source), context)
    assert requests == [key] and source.read_bytes() == b'unchanged input'


def test_failed_native_screen_construction_does_not_return_a_phantom_module(commands, monkeypatch):
    commands._screens = {}
    requested = []
    monkeypatch.setattr(commands, 'open_module', requested.append, raising=False)
    monkeypatch.setattr(app, 'app_is_visible', lambda key: True)
    assert app.MainWindow._open_easy_named_module(commands, 'mask') is None
    assert requested == ['mask'] and commands._screens == {}
