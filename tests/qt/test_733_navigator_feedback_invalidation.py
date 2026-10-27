"""Enter feedback and completed cards release stale software field frames."""
import pytest
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QImage, QRegion
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


def _hold_obsolete_square(page, backdrop, monkeypatch):
    from spacr.qt.widgets.ambient import _field_growing_regions_for_widget

    _field_growing_regions_for_widget(page.field, (), invalidate=True)
    live = backdrop._last_frame.copy()
    live.fill(QColor(0, 30, 100))
    obsolete = live.copy()
    obsolete.fill(QColor(210, 20, 180))
    producer = backdrop._producer_box[0]
    assert producer is not None
    monkeypatch.setattr(producer, 'latest', lambda: live)
    backdrop._last_frame = live
    center = backdrop.mapFromGlobal(page.path_edit.mapToGlobal(
        QPoint(page.path_edit.width() // 2, -80)))
    rectangle = QRect(center.x() - 25, center.y() - 25, 50, 50)
    assert backdrop.rect().contains(rectangle)
    held = backdrop._held_rows
    held.frame = obsolete
    held.region = QRegion(rectangle)
    held.fade = None
    backdrop.repaint()
    point = page.window().mapFromGlobal(backdrop.mapToGlobal(center))
    before = page.window().grab().toImage().convertToFormat(QImage.Format_RGBA8888)
    return held, point, before.pixelColor(point)


def test_feedback_only_transition_clears_held_square_without_mouse_input(
        qtbot, live_navigator, monkeypatch):
    window, page, backdrop = live_navigator
    page.field.set_cards([])
    held, point, before = _hold_obsolete_square(page, backdrop, monkeypatch)
    assert held.frame is not None
    page._feedback('No matching local input.', visible=False)
    qtbot.wait(30)
    assert held.frame is None and held.fade is None
    after = window.grab().toImage().pixelColor(point)
    assert abs(after.red() - before.red()) > 80
    assert not page.field.cards


def test_enter_card_completion_releases_obsolete_field_without_mouse_input(
        qtbot, live_navigator, monkeypatch):
    window, page, backdrop = live_navigator
    monkeypatch.setattr(window, '_easy_navigation_suggestions', lambda value: [
        ('measure', 'Measure', 'Review image measurements.', '')])
    page.path_edit.setText('detect features')
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    qtbot.waitUntil(lambda: bool(page.field.cards))
    assert page.field.cards[0]._key == 'measure'
    assert any(card._growing for card in page.field.cards)
    held, _point, _before = _hold_obsolete_square(page, backdrop, monkeypatch)
    qtbot.waitUntil(lambda: all(not card._growing for card in page.field.cards), timeout=2000)
    assert held.frame is None and held.fade is None
    assert all(card.isVisible() and card.isEnabled()
               and card._content_opacity.opacity() == 1 for card in page.field.cards)
