"""Organism Enter requests show cards; disabling Navigator preserves native work."""
import pytest
from PySide6.QtCore import Qt
from spacr.qt.organisms import ORGANISMS
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


@pytest.mark.parametrize(('key', 'query'),
    [(key, data['name']) for key, data in ORGANISMS.items()] +
    [('toxoplasma', 'TOXOPLASMA GONDII'), ('toxoplasma', 'How does T. gondii replicate?')])
def test_organism_enter_shows_card_without_direct_navigation(
        qtbot, private_ai_profile, key, query):
    from spacr.qt.widgets.home import _EasyHomePage
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_show_alpha(True)
    preferences._set_show_alpha_species(True)
    preferences.set_reduce_motion(True)
    page = _EasyHomePage()
    qtbot.addWidget(page)
    commands, opened = [], []
    page.command_requested.connect(commands.append)
    page.module_requested.connect(lambda *args: opened.append(args))
    page.show()
    page.path_edit.setText(query)
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    assert key in [card._key for card in page.field.cards]
    assert not commands and not opened
    assert page._workflow is None
    page.close()


def test_help_uncheck_retains_same_mask_screen_and_settings(qtbot, private_ai_profile):
    from spacr.qt.app import MainWindow
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.show()
    window._open_easy_named_module('mask')
    screen = window._screens['mask']
    screen.apply_settings_dict({'number_of_organelles': 2, 'organelle_channel': 0})
    model = screen._settings_model
    assert window._stack.currentWidget() is screen and window._act_easy_mode.isChecked()
    window._act_easy_mode.trigger()
    assert not preferences.get_easy_mode() and not window._act_easy_mode.isChecked()
    assert window._stack.currentWidget() is screen
    assert screen._settings_model is model
    assert model.collect()['number_of_organelles'] == 2
    assert model.collect()['organelle_channel'] == 0
    window.close()
