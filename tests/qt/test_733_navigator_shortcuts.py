"""Navigator shortcuts use existing actions and keep external applications click-only."""
import pytest
from PySide6.QtCore import Qt, QUrl

from spacr.qt import app, preferences, theme
from spacr.qt.widgets.home import _EasyHomePage
from tests.qt.test_733_easy_commands import commands as commands
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


@pytest.fixture
def shortcut_page(commands, qtbot, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app, 'app_is_visible', lambda key: True)
    monkeypatch.setattr(preferences, 'maturity_is_visible', lambda stage: True)
    page = _EasyHomePage(commands)
    commands.setCentralWidget(page)
    commands._startup = page
    commands._easy_navigation_suggestions = lambda value: app.MainWindow._easy_navigation_suggestions(commands, value)
    page.command_requested.connect(commands._open_easy_command)
    calls = []
    page.module_requested.connect(lambda key, path: app.MainWindow._open_easy_suggestion(commands, key, path))
    commands._open_organism_module = lambda key: calls.append(('organism', key))
    commands._check_for_updates = lambda: calls.append(('update',))
    monkeypatch.setattr(theme, 'enable_spaceout', lambda: calls.append(('spaceout',)))
    monkeypatch.setattr(app, '_easy_shortcut_folder', lambda command: tmp_path / command)
    return page, calls


@pytest.mark.parametrize('phrase,command', [
    ('update spacr', 'updater'), ('uppdate spacr', 'updater'), ('spacr updates', 'updater'),
    ('pypi', 'pypi'), ('pip', 'pypi'), ('pip install spacr', 'pypi'), ('install spacr', 'pypi'),
    ('conda', 'conda'), ('conda install spacr', 'conda'), ('conda-forge', 'conda'),
    ('stats', 'statistics'), ('spacr stats', 'statistics'), ('statistics', 'statistics'),
    ('logs', 'logs'), ('spacr logs', 'logs'), ('cach', 'cache'), ('cache', 'cache'),
    ('models', 'models'), ('spacr models', 'models'), ('docker', 'docker'),
    ('tutorials', 'tutorials'), ('spacr tutorials', 'tutorials'), ('nightly', 'nightly'),
])
def test_real_enter_shortcuts_use_existing_actions_without_module_launch(
        commands, shortcut_page, qtbot, tmp_path, phrase, command):
    page, calls = shortcut_page
    page.path_edit.setText(phrase)
    assert not calls and not commands.urls
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    if command == 'updater':
        assert calls == [('update',)]
    elif command in app._EASY_SHORTCUT_URLS:
        assert commands.urls == [app._EASY_SHORTCUT_URLS[command]]
    else:
        assert commands.urls == [QUrl.fromLocalFile(str(tmp_path / command)).toString()]
    if command == 'models':
        assert {card._key for card in page.field.cards} >= {'model_zoo', '__easy_command__:Models'}
    assert not commands.opened and not commands.navigated


@pytest.mark.parametrize('locale', tuple(app._EASY_SHORTCUT_LOCALIZED))
def test_all_localized_shortcut_nouns_and_actions_keep_the_same_targets(monkeypatch, tmp_path, locale):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('SPACR_LANGUAGE', locale)
    for command, phrases in app._EASY_SHORTCUT_LOCALIZED[locale].items():
        for phrase in phrases:
            assert app._easy_shortcut_match(phrase) == command, (locale, command, phrase)
            assert app._easy_shortcut_match(phrase.upper() + '!') == command


@pytest.mark.parametrize('key,phrase', [('__easy_command__:Spaceout', 'Spaceout'), ('parascape', 'ParaScape')])
def test_spaceout_and_parascape_require_an_actual_card_click(
        commands, shortcut_page, qtbot, key, phrase):
    page, calls = shortcut_page
    page.resize(1000, 800)
    page.show()
    page.path_edit.setText(phrase)
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    qtbot.waitUntil(lambda: bool(page.field.cards), timeout=3000)
    assert [card._key for card in page.field.cards] == [key]
    assert not calls and not commands.urls and not commands.opened
    card = page.field.cards[0]
    qtbot.waitUntil(lambda: not card._growing, timeout=4000)
    qtbot.mouseClick(card, Qt.LeftButton)
    assert calls == ([('spaceout',)] if key.startswith('__easy') else [('organism', 'parascape')])
    if key.startswith('__easy'):
        assert commands.refreshed == [{'spaceout'}]
    else:
        assert card._stage == 'alpha'


def test_folder_shortcuts_resolve_configured_application_paths(monkeypatch, tmp_path):
    import spacr.logging_util as logging_util
    from spacr.qt import app as app_module

    monkeypatch.setattr(logging_util, '_spacr_home', lambda: tmp_path / 'service')
    monkeypatch.setenv('SPACR_LOG_DIR', str(tmp_path / 'configured logs'))
    monkeypatch.setenv('SPACR_MODEL_CACHE', str(tmp_path / 'configured models'))
    assert app_module._easy_shortcut_folder('logs') == tmp_path / 'configured logs'
    assert app_module._easy_shortcut_folder('cache') == tmp_path / 'service' / 'cache'
    assert app_module._easy_shortcut_folder('models') == tmp_path / 'configured models'


@pytest.mark.parametrize('value', ['/data/models', './stats', 'dump.sql', 'open logs.csv', r'C:\cache\file.npy'])
def test_shortcut_aliases_do_not_capture_real_paths_or_filenames(value):
    assert app._easy_shortcut_match(value) is None


def test_existing_keyword_folder_is_not_a_typed_command_but_explicit_card_remains_an_action(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'models').mkdir()
    assert app._easy_shortcut_match('models') is None
    assert app._easy_shortcut_match('Models', explicit=True) == 'models'


def test_parascape_explicit_route_keeps_alpha_guard(commands, monkeypatch):
    called = []
    commands._open_organism_module = called.append
    monkeypatch.setattr(preferences, 'maturity_is_visible', lambda stage: stage != 'alpha')
    app.MainWindow._open_easy_suggestion(commands, 'parascape', '')
    assert not called and not commands.opened


@pytest.mark.parametrize('locale', tuple(app._EASY_SHORTCUT_LOCALIZED))
def test_localized_statistics_action_takes_real_enter_precedence(
        commands, shortcut_page, qtbot, monkeypatch, locale):
    monkeypatch.setenv('SPACR_LANGUAGE', locale)
    page, calls = shortcut_page
    phrase = app._EASY_SHORTCUT_LOCALIZED[locale]['statistics'][1]
    page.path_edit.setText(phrase)
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    assert commands.urls == [app._EASY_SHORTCUT_URLS['statistics']]
    assert not commands.opened and not calls
