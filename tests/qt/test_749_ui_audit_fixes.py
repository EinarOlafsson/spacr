"""Low-risk findings of the core-module UI audit, each fixed at its source."""
import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialogButtonBox

from spacr.condition_annotations import new_definition, source_context
from spacr.qt import app, walkthrough
from spacr.qt.i18n_module_summaries import module_summary
from spacr.qt.screens.app_screen import _FreeRamDialog
from spacr.qt.widgets import condition_annotation_dialog as conditions
from spacr.qt.widgets.home import _EasyHomePage
from tests.qt.test_733_easy_commands import commands as commands
from tests.qt.test_ram_guard_dialog import _Psutil
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


@pytest.mark.parametrize('key', ['mask', 'measure', 'make_masks', 'umap'])
def test_the_first_walkthrough_card_ends_the_summary_before_the_escape_hint(key):
    body = walkthrough.build_steps(key)[0].body
    summary = next(row[2] for row in app.APPS if row[0] == key).rstrip()
    assert body.startswith(summary)
    assert body[len(summary)] in '.!?…。'
    assert body[len(summary):].lstrip('.!?…。').startswith(' Press Esc')


@pytest.mark.parametrize('query', ['Measure', 'mask', 'segment images'])
def test_navigator_card_descriptions_use_the_reviewed_german_summaries(monkeypatch, query):
    monkeypatch.setenv('SPACR_LANGUAGE', 'de')
    rows = app._easy_module_search(query)
    assert rows
    english = {row[0]: row[2] for row in app.APPS}
    for key, _title, description, _path in rows:
        if key in english:
            reviewed = module_summary(key, english[key], 'de')
            assert reviewed != english[key]
            assert description == reviewed


def test_navigator_descriptions_stay_english_in_english(monkeypatch):
    monkeypatch.setenv('SPACR_LANGUAGE', 'en')
    english = {row[0]: row[2] for row in app.APPS}
    rows = app._easy_module_search('Measure')
    assert [(key, description) for key, _t, description, _p in rows if key == 'measure'] == [
        ('measure', english['measure'])]


def test_tab_moves_from_the_composer_to_the_cards_before_the_side_panel(
        commands, qtbot, monkeypatch):
    monkeypatch.setattr(app, 'app_is_visible', lambda key: True)
    page = _EasyHomePage(commands)
    commands.setCentralWidget(page)
    commands._startup = page
    commands._easy_navigation_suggestions = (
        lambda value: app.MainWindow._easy_navigation_suggestions(commands, value))
    page.command_requested.connect(commands._open_easy_command)
    commands.show()
    page.path_edit.setText('segment images')
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    qtbot.waitUntil(lambda: bool(page.field.cards), timeout=3000)
    cards = page.field.cards
    chain = []
    widget = page.path_edit.protocols.nextInFocusChain()
    while widget is not None and len(chain) < 40 and widget is not page.path_edit.protocols:
        chain.append(widget)
        widget = widget.nextInFocusChain()
    positions = [chain.index(card) for card in cards]
    assert positions == sorted(positions)
    assert positions[-1] < chain.index(page.field.details)
    assert positions[0] == 0


def test_the_free_ram_dialog_orders_its_buttons_for_the_platform_and_defaults_to_cancel(qapp):
    dialog = _FreeRamDialog(None, psutil_module=_Psutil(), confirm=lambda names: False)
    box = dialog.findChild(QDialogButtonBox)
    assert box is not None
    assert box.buttonRole(dialog.close_button) == QDialogButtonBox.AcceptRole
    cancel = [button for button in box.buttons() if button is not dialog.close_button]
    assert len(cancel) == 1 and box.buttonRole(cancel[0]) == QDialogButtonBox.RejectRole
    assert cancel[0].isDefault() and not dialog.close_button.isDefault()
    dialog.deleteLater()


def test_the_selection_label_dialog_uses_a_platform_button_box(qtbot):
    frame = pd.DataFrame({'well': ['A', 'B'], 'value': [1, 2]})
    source = source_context()
    dialog = conditions._SelectionAnnotationDialog(
        None, frame, source, new_definition(frame, source), [0])
    qtbot.addWidget(dialog)
    box = dialog.findChild(QDialogButtonBox)
    assert box is not None
    assert box.buttonRole(dialog.apply_button) == QDialogButtonBox.AcceptRole
    roles = sorted(int(box.buttonRole(button).value) for button in box.buttons())
    assert roles == sorted([int(QDialogButtonBox.AcceptRole.value),
                            int(QDialogButtonBox.RejectRole.value)])


def test_a_dock_heading_takes_keyboard_focus_and_folds_with_space_and_enter(qtbot):
    from spacr.qt.widgets.dock import Dock, SectionHeader

    rows = [('mask', 'Mask', 'Masks', 'Core'), ('make_masks', 'Make Masks', 'Draw', 'Tools'),
            ('umap', 'Image UMAP', 'Embed', 'Tools')]
    dock = Dock(rows)
    qtbot.addWidget(dock)
    dock.show()
    header = next(h for h in dock.findChildren(SectionHeader) if h.section == 'Tools')
    assert header.focusPolicy() & Qt.TabFocus
    header.setFocus()
    qtbot.waitUntil(header.hasFocus, timeout=2000)
    assert header.property('hovered')
    before = dock.is_open('Tools') if hasattr(dock, 'is_open') else 'Tools' in dock._open
    qtbot.keyClick(header, Qt.Key_Space)
    after = dock.is_open('Tools') if hasattr(dock, 'is_open') else 'Tools' in dock._open
    assert after != before
    qtbot.keyClick(header, Qt.Key_Return)
    again = dock.is_open('Tools') if hasattr(dock, 'is_open') else 'Tools' in dock._open
    assert again == before


@pytest.mark.parametrize('theme_name', ['light', 'dark'])
@pytest.mark.parametrize('name,role', [('DangerButton', 'negative'), ('PrimaryButton', 'positive'),
                                       ('', 'negative'), ('', 'positive')])
def test_a_disabled_run_or_stop_button_drops_its_action_colour(qtbot, theme_name, name, role):
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QPushButton, QWidget

    from spacr.qt import theme

    palette = theme.palette_for(theme_name)
    host = QWidget()
    host.setStyleSheet(theme.stylesheet(theme_name))
    qtbot.addWidget(host)
    button = QPushButton('Stop' if role == 'negative' else 'Run', host)
    if name:
        button.setObjectName(name)
    button.setProperty('buttonActionRole', role)
    button.resize(120, 40)
    host.resize(140, 60)
    host.show()
    qtbot.waitExposed(host)

    def colours():
        image = button.grab().toImage()
        return {QColor(image.pixel(x, y)).name() for x in range(image.width())
                for y in range(image.height())}

    action = palette['error'] if role == 'negative' else palette['button_accent']
    assert action.lower() in colours()
    button.setEnabled(False)
    host.style().unpolish(button)
    host.style().polish(button)
    assert action.lower() not in colours()


@pytest.mark.parametrize('key,title', [('umap', 'Point Image UMAP at some data'),
                                       ('mask', 'Point Mask Generation at some data')])
def test_the_empty_state_headline_keeps_the_module_name_as_written(qtbot, key, title):
    from PySide6.QtWidgets import QLabel

    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(key)
    qtbot.addWidget(screen)
    card = screen._empty_state_card
    assert card is not None
    assert title in [label.text() for label in card.findChildren(QLabel)]


@pytest.mark.parametrize('theme_name', ['light', 'dark'])
def test_the_live_preview_toggle_wears_the_disclosure_surface(qtbot, theme_name):
    from collections import Counter

    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QToolButton

    from spacr.qt import theme
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen('mask')
    qtbot.addWidget(screen)
    screen.setStyleSheet(theme.stylesheet(theme_name))
    from spacr.qt import settings_search

    bar = getattr(screen, '_settings_search', None) or settings_search.install(screen)
    assert bar is not None
    toggle = QToolButton()
    toggle.setObjectName('SettingsPreviewToggle')
    toggle.setText('Live preview')
    toggle.setCheckable(True)
    bar.add_trailing_widget(toggle)
    screen.resize(1600, 900)
    screen.show()
    qtbot.waitExposed(screen)
    qtbot.wait(100)

    def surface(widget):
        rect = widget.rect().adjusted(3, 3, -3, -3)
        image = screen.grab(rect.translated(widget.mapTo(screen, rect.topLeft()) - rect.topLeft())).toImage()
        return Counter(QColor(image.pixel(x, y)).name() for x in range(image.width())
                       for y in range(image.height())).most_common(1)[0][0]

    assert surface(toggle) == surface(bar._disclosure)
