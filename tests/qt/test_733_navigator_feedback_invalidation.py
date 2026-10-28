"""Enter feedback and completed cards release stale software field frames."""
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


@pytest.fixture
def live_navigator(qtbot, private_ai_profile, monkeypatch):
    from spacr.qt.app import MainWindow

    monkeypatch.setenv('SPACR_NO_GL', '1')
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_ambient_enabled(True)
    preferences.set_reduce_motion(False)
    preferences.set_theme('dark')
    preferences.set_ambient_theme('data_art_impulse_lens')
    preferences.apply_preferences_to_app(QApplication.instance())
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1100, 820)
    window.show()
    qtbot.waitUntil(lambda: window.window_backdrop() is not None)
    backdrop = window.window_backdrop()
    qtbot.waitUntil(lambda: backdrop._last_frame is not None)
    qtbot.mouseMove(window, QPoint(4, window.height() - 4))
    try:
        yield window, window._startup, backdrop
    finally:
        window.close()


def _assert_overlay_is_live(page, backdrop):
    held = backdrop._held_rows
    point = backdrop.mapFromGlobal(page.path_edit.mapToGlobal(
        QPoint(page.path_edit.width() // 2, -80)))
    assert not held._covered().contains(point)
    assert not held.region.contains(point)
    assert not backdrop._art_input._growing_regions


def test_feedback_only_transition_has_no_field_physics_without_mouse_input(
        qtbot, live_navigator, monkeypatch):
    window, page, backdrop = live_navigator
    from spacr.qt.widgets import ambient

    def unexpected(*args, **kwargs):
        pytest.fail('Navigator changed the shared ambient field')

    monkeypatch.setattr(ambient, '_field_growing_regions_for_widget', unexpected)
    monkeypatch.setattr(ambient, 'field_ripple_for_widget', unexpected)
    page.field.set_cards([])
    page._feedback('No matching local input.', visible=False)
    before = page.field.rim.grab().toImage()
    qtbot.mouseClick(page.path_edit, Qt.LeftButton)
    qtbot.wait(1900)
    assert page.field.rim.grab().toImage() == before
    assert not hasattr(page.field.rim, '_timer')
    assert not hasattr(page.path_edit, '_gravity_clock')
    assert not page.field.cards
    _assert_overlay_is_live(page, backdrop)


def test_enter_card_growth_preserves_selected_theme_and_live_field(
        qtbot, live_navigator, monkeypatch):
    window, page, backdrop = live_navigator
    backdrop.set_background_color(QColor('#345678'))
    original = (backdrop.background_color(), backdrop._background_explicit, backdrop.backdrop())
    producer = backdrop._producer_box[0]
    monkeypatch.setattr(window, '_easy_navigation_suggestions', lambda value: [
        ('measure', 'Measure', 'Review image measurements.', '')])
    page.path_edit.setText('detect features')
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    qtbot.waitUntil(lambda: bool(page.field.cards))
    assert page.field.cards[0]._key == 'measure'
    assert any(card._growing for card in page.field.cards)
    qtbot.waitUntil(lambda: all(not card._growing for card in page.field.cards), timeout=2000)
    qtbot.wait(1900)
    assert all(card.isVisible() and card.isEnabled()
               and card._content_opacity.opacity() == 1 for card in page.field.cards)
    _assert_overlay_is_live(page, backdrop)
    assert (backdrop.background_color(), backdrop._background_explicit, backdrop.backdrop()) == original
    assert backdrop._producer_box[0] is producer
    assert not window.property('spacrEasySearchRippleOnly')
    page.hide()
    page.show()
    assert (backdrop.background_color(), backdrop._background_explicit, backdrop.backdrop()) == original


def test_real_no_answer_does_not_rearm_a_held_placeholder_square(
        qtbot, live_navigator):
    window, page, backdrop = live_navigator
    page.field.set_cards([])
    page.path_edit.setText('unrecognized input')
    page._requested = page.path_edit.text()
    message = 'No local module matches this input.'
    page._show_result(page._input_generation, {
        'path': page._requested, 'actions': [], 'message': message,
    })
    qtbot.wait(1900)
    assert not page.field.cards
    assert page.field.details.toPlainText() == message
    assert page.path_edit.accessibleDescription() == message
    _assert_overlay_is_live(page, backdrop)
