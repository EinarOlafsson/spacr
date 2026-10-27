"""All Navigator choices reveal at the requested cadence and Home clears exploration."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


def test_every_card_reveals_with_200ms_start_spacing_and_500ms_duration(
        qtbot, private_ai_profile):
    from spacr.qt.widgets.home import _EasyHomePage
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_ambient_enabled(True)
    preferences.set_reduce_motion(False)
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.resize(1960, 820)
    page.show()
    keys = ['mask', 'make_masks', 'measure', 'db_browser', 'graph_builder',
            'gate_editor', 'regression', 'annotate']
    page.field.set_cards([(key, key, '', '') for key in keys])
    cards = page.field.cards
    assert len(page.field._animations) == len(cards)
    for index, (card, group) in enumerate(zip(cards, page.field._animations)):
        assert card._growing and not card.isEnabled()
        assert group.animationAt(0).duration() == index * 200
        assert group.animationAt(1).duration() == 500
        assert card._content_opacity.opacity() == 0
    qtbot.waitUntil(lambda: 0 < cards[4]._content_opacity.opacity() < 1, timeout=2000)
    assert cards[4]._growing
    qtbot.waitUntil(lambda: all(not card._growing for card in cards), timeout=3000)
    assert all(card._content_opacity.opacity() == 1 and card.isEnabled() for card in cards)
    page.close()


def test_returning_home_discards_graph_input_and_stale_results(qtbot, private_ai_profile):
    from spacr.qt.app import MainWindow
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.show()
    page = window._startup
    page.path_edit.setText('/local/project')
    page._requested = page.path_edit.text()
    result = dict(path=page.path_edit.text(), message='', context={},
                  actions=[('mask', 'Mask', '', '/local/project')])
    page._show_result(page._input_generation, result)
    card = page.field.cards[0]
    qtbot.mouseClick(card, Qt.LeftButton)
    assert page._workflow is not None
    generation = page._input_generation
    other = QWidget(window)
    window._stack.addWidget(other)
    window._stack.setCurrentWidget(other)
    window._on_nav_selected('__home__')
    assert window._stack.currentWidget() is page
    assert page._workflow is None and not page.field.cards
    assert page.path_edit.text() == '' and page.path_edit.isVisible()
    assert not page.field.details.toPlainText()
    assert page._last_result is None and page._settings_overrides == {}
    assert not page._queued_inspection and not page._debounce.isActive()
    page._show_result(generation, result)
    assert not page.field.cards and page._workflow is None
    window.close()
