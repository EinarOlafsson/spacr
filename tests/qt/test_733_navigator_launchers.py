"""Installed Navigator aliases preserve the normal GUI launch protocol."""
import ast
import sys
from pathlib import Path

import pytest

from spacr.qt import app
from tests.qt.test_cov_qt_app import launched
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


def test_navigator_commands_use_the_existing_console_entrypoint():
    tree = ast.parse((Path(__file__).resolve().parents[2] / 'setup.py').read_text())
    setup = next(node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'setup')
    entries = ast.literal_eval(next(item.value for item in setup.keywords if item.arg == 'entry_points'))
    scripts = dict(row.split('=', 1) for row in entries['console_scripts'])
    for command in ('spacr', 'spacr-qt', 'spacr-nightly', 'spacr-navigator', 'spacr-nav'):
        assert scripts[command] == 'spacr.qt:run'
    for command in ('spaceout', 'spaceout-navigator', 'spaceout-nav'):
        assert scripts[command] == 'spacr.qt.spaceout:main'


@pytest.mark.parametrize('executable', [
    '/env/bin/spacr-navigator', '/env/bin/spacr-nav',
    '/env/bin/spaceout-navigator', '/env/bin/spaceout-nav',
    r'C:\env\Scripts\SPACEOUT-NAVIGATOR.EXE', r'C:\env\Scripts\spaceout-nav-script.py',
    r'C:\env\Scripts\SPACR-NAVIGATOR.EXE', r'C:\env\Scripts\spacr-nav.exe',
    r'C:\env\Scripts\spacr-navigator-script.py', r'C:\env\Scripts\SPACR-NAV-script.py',
])
def test_aliases_enable_persisted_navigator_home(private_ai_profile, executable):
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(False)
    assert app._navigator_launch_target(None, executable) == '__home__'
    assert preferences.get_easy_mode()


@pytest.mark.parametrize('executable', ['spacr', 'spacr-qt', 'spacr-navigation.exe'])
def test_other_launchers_leave_the_preference_and_target_untouched(private_ai_profile, executable):
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(False)
    assert app._navigator_launch_target('measure', executable) == 'measure'
    assert not preferences.get_easy_mode()


def test_explicit_module_wins_while_navigator_preference_is_enabled(private_ai_profile):
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(False)
    assert app._navigator_launch_target('mask', 'spacr-nav') == 'mask'
    assert preferences.get_easy_mode()


@pytest.mark.parametrize('command', ['spacr-navigator', 'spacr-nav', 'spaceout-navigator', 'spaceout-nav'])
def test_version_only_alias_never_changes_preferences(private_ai_profile, monkeypatch, capsys, command):
    import spacr.qt as qt
    from spacr.version import get_version

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(False)
    monkeypatch.setattr(sys, 'argv', [command, '--version'])
    if command.startswith('spaceout'):
        from spacr.qt import spaceout, theme
        was_spaceout = theme.spaceout_enabled()
        try:
            assert spaceout.main() == 0
        finally:
            theme.enable_spaceout() if was_spaceout else theme.disable_spaceout()
    else:
        assert qt.run() == 0
    assert get_version() in capsys.readouterr().out
    assert not preferences.get_easy_mode()


@pytest.mark.parametrize('command,args', [
    ('spacr-navigator', []), ('spacr-nav', ['--no-setup', '--fresh']),
    ('spaceout-navigator', []), ('spaceout-nav', ['--no-setup', '--fresh']),
])
def test_real_launch_applies_alias_after_setup_and_keeps_launch_flags(
        launched, private_ai_profile, monkeypatch, command, args):
    from spacr.qt.widgets import setup_slides
    from spacr.qt.widgets.home import _EasyHomePage

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(False)
    monkeypatch.setenv('SPACR_NO_BACKDROP', '1')
    monkeypatch.setattr(sys, 'argv', [command, *args])
    setup_calls = []

    def setup(_parent):
        setup_calls.append(True)
        preferences.set_easy_mode(False)
        return True

    monkeypatch.setattr(setup_slides, 'open_setup_if_needed', setup)
    if command.startswith('spaceout'):
        import spacr.qt as qt
        from spacr.qt import spaceout, theme
        was_spaceout = theme.spaceout_enabled()
        monkeypatch.setattr(qt, 'run', app.launch)
        try:
            assert spaceout.main(args) == 0
            assert theme.spaceout_enabled()
        finally:
            theme.enable_spaceout() if was_spaceout else theme.disable_spaceout()
    else:
        assert app.launch(args) == 0
    window = launched['window']()
    assert preferences.get_easy_mode()
    assert window._stack.currentWidget() is window._startup
    assert isinstance(window._startup, _EasyHomePage)
    assert app._OPEN_FRESH[0] == ('--fresh' in args)
    assert setup_calls == ([] if '--no-setup' in args else [True])


def test_fresh_profile_defaults_to_navigator_home_and_startup_checkbox(qtbot, private_ai_profile):
    from spacr.qt import preferences
    from spacr.qt.widgets.home import _EasyHomePage
    from spacr.qt.widgets.setup_dialog import SetupDialog
    from spacr.qt.widgets.setup_slides import SetupSlides

    assert not preferences._settings().contains(preferences._KEY_EASY_MODE)
    assert preferences.get_easy_mode()
    page = app.make_home_page()
    qtbot.addWidget(page)
    assert isinstance(page, _EasyHomePage)
    dialog = SetupDialog()
    slides = SetupSlides()
    qtbot.addWidget(dialog)
    qtbot.addWidget(slides)
    assert dialog._editors['easy_mode'].isChecked()
    assert slides._editors['easy_mode'].isChecked()
    assert not preferences._settings().contains(preferences._KEY_EASY_MODE)


@pytest.mark.parametrize('value', [False, 'false', '0', True, 'true', '1'])
def test_saved_navigator_choices_override_the_new_default(private_ai_profile, value):
    from spacr.qt import preferences

    preferences._settings().setValue(preferences._KEY_EASY_MODE, value)
    expected = value in (True, 'true', '1')
    assert preferences.get_easy_mode() is expected
    assert preferences._settings().value(preferences._KEY_EASY_MODE) == value
