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


@pytest.fixture
def navigator_window(qtbot, private_ai_profile):
    from spacr.qt.app import MainWindow

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    preferences.set_ambient_enabled(False)
    windows = []

    def build(scale=1.0, size=(1600, 1000), dock_collapsed=True):
        preferences.set_font_scale(scale)
        preferences._settings().setValue('prefs/navigator_dock_collapsed', dock_collapsed)
        window = MainWindow(initial_app='__home__')
        qtbot.addWidget(window)
        windows.append(window)
        window._tour_timer.stop()
        window._consent_timer.stop()
        window.resize(*size)
        window.show()
        qtbot.waitUntil(lambda: window._startup.field.width() > 600)
        return window

    yield build
    for window in windows:
        window.close()


def test_the_protocols_selector_is_left_out_of_the_field_fade(navigator_window):
    from spacr.qt.widgets.field_fade import fades
    from spacr.qt.widgets.home import _NavigatorProtocolChoice

    window = navigator_window()
    page = window._startup
    page.path_edit.protocols.setChecked(True)
    selectors = page.findChildren(_NavigatorProtocolChoice)
    assert selectors
    assert not any(fades(selector) for selector in selectors)


def test_the_widgets_panel_folds_when_the_composer_has_no_room_and_returns(
        navigator_window, qtbot):
    window = navigator_window(scale=2.0, size=(2400, 1300))
    page = window._startup
    split = page._navigator_aside_split
    qtbot.waitUntil(lambda: not split.is_collapsed('Widgets'), timeout=3000)
    window.resize(1366, 768)
    qtbot.waitUntil(lambda: split.is_collapsed('Widgets'), timeout=3000)
    panel = page.path_edit
    assert panel.width() >= 600
    window.resize(2400, 1300)
    qtbot.waitUntil(lambda: not split.is_collapsed('Widgets'), timeout=3000)


def test_a_fold_the_person_undoes_stays_undone_at_the_same_width(navigator_window, qtbot):
    window = navigator_window(scale=2.0, size=(1366, 768))
    split = window._startup._navigator_aside_split
    qtbot.waitUntil(lambda: split.is_collapsed('Widgets'), timeout=3000)
    split.set_collapsed('Widgets', False, by_user=True)
    window.resize(1370, 770)
    qtbot.wait(200)
    assert not split.is_collapsed('Widgets')


def test_a_module_card_explanation_is_headed_by_its_module(navigator_window, qtbot):
    window = navigator_window()
    page = window._startup
    page.path_edit.setText('segment images')
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    qtbot.waitUntil(lambda: bool(page.field.cards), timeout=3000)
    rows = {key: title for key, title, _d, _p in app.MainWindow._easy_navigation_suggestions(
        window, 'segment images')}
    for card in page.field.cards:
        if card._key in rows:
            assert card._explanation.startswith('<b>' + rows[card._key] + '</b>')


@pytest.mark.parametrize('width', [800, 1800])
def test_the_settings_column_keeps_a_working_width(qtbot, width):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen('mask')
    qtbot.addWidget(screen)
    screen.resize(width, 800)
    screen.show()
    qtbot.waitExposed(screen)
    body = screen._body_splitter
    from spacr.qt.preferences import scaled_px

    qtbot.waitUntil(lambda: body.sizes()[0] >= min(scaled_px(360), int(body.width() * 0.55)) - 8,
                    timeout=3000)
    settings, runtime = body.sizes()
    if width >= 1800:
        assert runtime > settings


def test_a_long_empty_state_title_wraps_instead_of_widening_the_form(qtbot):
    from PySide6.QtGui import QFontMetrics
    from PySide6.QtWidgets import QLabel

    from spacr.qt.widgets.empty_state import EmptyState

    title = 'Point a module with a very long name indeed at some data'
    card = EmptyState(title=title, subtitle='Drop a folder of images here.')
    qtbot.addWidget(card)
    label = next(lbl for lbl in card.findChildren(QLabel) if lbl.text() == title)
    assert label.wordWrap()
    assert label.minimumSizeHint().width() < QFontMetrics(label.font()).horizontalAdvance(title)


def test_the_description_panel_stays_between_the_dock_and_the_widgets_panel(
        navigator_window, qtbot):
    from PySide6.QtCore import QPoint

    window = navigator_window(size=(1600, 1000))
    page = window._startup
    field = page.field
    split = page._navigator_aside_split
    assert not split.is_collapsed('Widgets')
    aside_left = field.mapFromGlobal(page._navigator_aside.mapToGlobal(QPoint(0, 0))).x()
    qtbot.waitUntil(lambda: field.details.geometry().right() < aside_left, timeout=3000)
    window._dock_edge.set_collapsed(False)
    edge = window._dock_edge
    dock_right = field.mapFromGlobal(edge.mapToGlobal(QPoint(edge.width(), 0))).x()
    qtbot.waitUntil(lambda: field.details.geometry().left() >= dock_right, timeout=3000)
    assert field.details.geometry().right() < aside_left
    window._dock_edge.set_collapsed(True)
    qtbot.waitUntil(lambda: field.details.geometry().left() < dock_right, timeout=3000)
    split.set_collapsed('Widgets', True, by_user=True)
    qtbot.waitUntil(lambda: field.details.geometry().right() > aside_left, timeout=3000)


def test_the_dock_folds_when_the_content_has_no_room_and_returns(
        qtbot, private_ai_profile):
    from spacr.qt.app import MainWindow

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(False)
    preferences.set_reduce_motion(True)
    preferences.set_font_scale(2.0)
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(2600, 1400)
    window.show()
    qtbot.waitExposed(window)
    edge = window._dock_edge
    edge.set_collapsed(False)
    window.resize(1366, 768)
    qtbot.waitUntil(edge.is_collapsed, timeout=3000)
    window.resize(2600, 1400)
    qtbot.waitUntil(lambda: not edge.is_collapsed(), timeout=3000)
    window.resize(1366, 768)
    qtbot.waitUntil(edge.is_collapsed, timeout=3000)
    edge.set_collapsed(False)
    window.resize(1370, 770)
    qtbot.wait(200)
    assert not edge.is_collapsed()
    edge.set_collapsed(True)
    window.resize(2600, 1400)
    qtbot.wait(200)
    assert edge.is_collapsed()
    window.close()


def test_a_roomy_rectangle_a_large_zoom_outgrew_scrolls_instead_of_squeezing(
        qtbot, monkeypatch):
    """The Dock's automatic fold freed a roomy width; the screen still squeezed.

    A module screen's content fills its rectangle once the rectangle is
    roomy, and roomy was a fixed 1500 x 900. Every control grows with the
    Zoom, so at 200 % the same rectangle is cramped: the Dock folding to its
    edge strip took the content from a width that scrolled to one that filled
    1588 pixels with controls asking for more, and cut their text off.
    """
    from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

    from spacr.qt import preferences

    monkeypatch.setattr(preferences, 'get_font_scale', lambda: 2.0)
    host = QWidget()
    qtbot.addWidget(host)
    outer = QVBoxLayout(host)
    outer.setContentsMargins(0, 0, 0, 0)
    content = QWidget()
    inner = QVBoxLayout(content)
    button = QPushButton('A control whose label is far too long to fit ' * 9)
    inner.addWidget(button)
    scroll = app._CrampedScroll(content, host)
    outer.addWidget(scroll)
    host.resize(1588, 1000)
    host.show()
    qtbot.waitExposed(host)
    need = content.minimumSizeHint().expandedTo(content.minimumSize()).width()
    assert scroll.width() > app._ROOMY_SCREEN[0] < need
    qtbot.waitUntil(lambda: content.width() >= need, timeout=3000)
    assert button.width() >= button.sizeHint().width() - 2
    host.close()


def test_tab_reaches_the_settings_form_before_the_console_and_run(qtbot, private_ai_profile):
    from PySide6.QtWidgets import QWidget

    from spacr.qt.app import MainWindow

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(False)
    preferences.set_reduce_motion(True)
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1600, 1000)
    window.show()
    qtbot.waitExposed(window)
    window._on_nav_selected('mask')
    qtbot.waitUntil(lambda: 'mask' in window._screens, timeout=10000)
    screen = window._screens['mask']
    form, side = screen._settings_panel, screen._runtime_wrap

    def chain():
        found, widget = [], window.nextInFocusChain()
        while widget is not None and widget is not window and len(found) < 20000:
            if widget.focusPolicy() & Qt.TabFocus:
                found.append(widget)
            widget = widget.nextInFocusChain()
        return found

    def ordered():
        widgets = chain()
        forms = [i for i, w in enumerate(widgets) if form.isAncestorOf(w)]
        sides = [i for i, w in enumerate(widgets) if side.isAncestorOf(w)]
        return bool(forms and sides) and max(forms) < min(sides)

    qtbot.waitUntil(ordered, timeout=5000)
    search = screen.findChild(QWidget, 'SettingsSearchInput')
    runs = [w for w in chain() if side.isAncestorOf(w) and w.objectName() == 'PrimaryButton']
    if search is not None and runs:
        assert chain().index(search) < chain().index(runs[0])
    window.close()


def test_every_walkthrough_caption_goes_through_the_translator(monkeypatch):
    from spacr.qt import i18n

    real = i18n.tr
    monkeypatch.setattr(i18n, 'tr', lambda text, *a, **k: '«' + real(text, *a, **k) + '»')
    steps = walkthrough.build_steps('mask')
    assert len(steps) >= 3
    for step in steps[:4]:
        assert step.title.startswith('«') and step.body.startswith('«')
    run = next(step for step in steps if 'Run starts' in step.body)
    assert 'console on the right' in run.body and 'template' in run.body


@pytest.mark.parametrize('phrase', ['segment cells', 'segment nuclei', 'nuclei segmentation'])
def test_cell_and_nucleus_phrasing_finds_the_segmentation_modules(phrase):
    assert {'mask', 'make_masks'} <= set(app._easy_task_keys(phrase))


def test_the_curate_paths_carry_their_captions_as_names(qtbot):
    from spacr.qt.screens.curate import CurateScreen

    screen = CurateScreen()
    qtbot.addWidget(screen)
    assert screen._mask_edit.accessibleName() == 'Mask'
    assert screen._tracks_edit.accessibleName() == 'Tracks'

@pytest.mark.parametrize('theme_name', ['light', 'dark'])
def test_an_unchecked_list_row_draws_an_outlined_box_not_a_filled_one(qtbot, theme_name):
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QListWidget, QListWidgetItem, QWidget

    from spacr.qt import theme

    palette = theme.palette_for(theme_name)
    host = QWidget()
    host.setStyleSheet(theme.stylesheet(theme_name))
    qtbot.addWidget(host)
    view = QListWidget(host)
    for checked in (Qt.Unchecked, Qt.Checked):
        item = QListWidgetItem('rhoptries', view)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(checked)
    view.resize(240, 120)
    host.resize(260, 140)
    host.show()
    qtbot.waitExposed(host)
    image = view.viewport().grab().toImage()
    ratio = image.devicePixelRatio()

    def box(row):
        rect = view.visualItemRect(view.item(row))
        colours = [QColor(image.pixel(int(x * ratio), int(y * ratio)))
                   for x in range(rect.left(), rect.left() + 22)
                   for y in range(rect.top() + 2, rect.bottom() - 1)]
        return {c.name() for c in colours}

    unchecked, checked = box(0), box(1)
    assert palette['surface_alt'].lower() in unchecked
    assert palette['accent'].lower() not in unchecked
    assert palette['accent'].lower() in checked


def test_a_toggle_is_a_24_pixel_target(qtbot):
    from spacr.qt.widgets.toggle import Toggle

    for toggle in (Toggle(), Toggle('Show older releases'), Toggle('Wrapped caption', word_wrap=True)):
        qtbot.addWidget(toggle)
        toggle.show()
        qtbot.waitExposed(toggle)
        assert toggle.minimumSizeHint().height() >= 24
        assert toggle.height() >= 24


@pytest.mark.parametrize('theme_name', ['light', 'dark'])
def test_a_home_panel_action_word_is_a_24_pixel_target(qtbot, theme_name):
    from PySide6.QtWidgets import QWidget

    from spacr.qt import theme
    from spacr.qt.widgets.home import Panel

    host = QWidget()
    host.setStyleSheet(theme.stylesheet(theme_name))
    qtbot.addWidget(host)
    panel = Panel('System', host)
    word = panel.add_action('Refresh', kind='safe')
    host.resize(300, 200)
    panel.resize(300, 200)
    host.show()
    qtbot.waitExposed(host)
    assert word.height() >= 24 and word.width() >= 24


def test_a_settings_field_action_button_is_a_centred_24_pixel_target(qtbot):
    from PySide6.QtWidgets import QLineEdit, QToolButton

    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen('mask')
    qtbot.addWidget(screen)
    screen.resize(1800, 1000)
    screen.show()
    qtbot.waitExposed(screen)
    button = screen.findChild(QToolButton, 'GenerateMasksFieldOverview')
    assert button is not None
    field = button.parentWidget()
    assert isinstance(field, QLineEdit)
    siblings = [b for b in field.findChildren(QToolButton) if b.isVisibleTo(field)]
    assert len(siblings) >= 2
    for each in siblings:
        assert each.height() >= 24 and each.width() >= 24
        assert abs(each.geometry().center().y() - field.rect().center().y()) <= 1
        assert each.geometry().top() >= 0 and each.geometry().bottom() <= field.rect().bottom()
    ordered = sorted(siblings, key=lambda b: b.x())
    for left, right in zip(ordered, ordered[1:]):
        assert left.geometry().right() < right.geometry().left()


def test_the_annotate_empty_state_scrolls_instead_of_overlapping_when_short(qtbot):
    from PySide6.QtWidgets import QLabel

    from spacr.qt.screens.annotate import AnnotateScreen

    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    screen.resize(1300, 560)
    screen.show()
    qtbot.waitExposed(screen)
    qtbot.wait(100)
    card = screen._empty_state
    assert screen._content_stack.currentWidget() is screen._empty_page
    assert screen._content_stack.height() < card.minimumSizeHint().height()
    labels = [label for label in card.findChildren(QLabel) if label.text()]
    button = card.cta_button
    bottom = max(label.geometry().bottom() for label in labels)
    assert button.geometry().top() > bottom
    assert card.height() >= card.minimumSizeHint().height()


def test_the_shortcut_and_organize_dialogs_use_platform_button_boxes(qtbot, tmp_path):
    from PySide6.QtWidgets import QMainWindow

    from spacr.qt import shortcuts
    from spacr.qt.widgets.organize_for_measure import OrganizeForMeasureDialog

    window = QMainWindow()
    qtbot.addWidget(window)
    keymap = shortcuts._KeymapDialog(window)
    qtbot.addWidget(keymap)
    box = keymap.findChild(QDialogButtonBox)
    assert box is not None
    assert box.buttonRole(keymap._btn_save) == QDialogButtonBox.AcceptRole
    assert box.buttonRole(keymap._btn_defaults) == QDialogButtonBox.ResetRole
    assert len(box.buttons()) == 3
    organize = OrganizeForMeasureDialog(str(tmp_path))
    qtbot.addWidget(organize)
    box = organize.findChild(QDialogButtonBox)
    assert box is not None
    assert box.buttonRole(organize.apply_button) == QDialogButtonBox.AcceptRole
    assert box.buttonRole(organize.cancel_button) == QDialogButtonBox.RejectRole
