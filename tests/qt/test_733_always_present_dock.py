"""Legacy dock visibility values never remove the native collapse control."""
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLineEdit, QWidget

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


@pytest.mark.parametrize('navigator', [False, True])
@pytest.mark.parametrize('legacy_mode', ['hidden', 'locked'])
def test_dock_stays_present_and_opens_without_replacing_active_module(
        qtbot, private_ai_profile, navigator, legacy_mode):
    from spacr.qt.app import MainWindow

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(navigator)
    preferences.set_easy_mode_dock(False)
    preferences.set_dock_mode(legacy_mode)
    preferences.set_dock_width(340)
    preferences.set_reduce_motion(True)
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    module = QWidget(window._stack)
    value = QLineEdit('unsaved module setting', module)
    window._stack.addWidget(module)
    window._stack.setCurrentWidget(module)
    window.show()
    edge = window._dock_edge
    try:
        window.apply_dock_mode('hidden')
        assert window._dock_mode == 'locked'
        assert edge.isVisible()
        assert edge.is_collapsed() == navigator
        assert window._act_all_apps.isEnabled()
        assert window._sidebar.width() == window._sidebar.column_width()
        width = window._sidebar.width()
        window.toggle_app_drawer()
        assert edge.isVisible() and not edge.is_collapsed()
        assert window._dock_slot.isVisible()
        qtbot.mouseClick(edge, Qt.LeftButton)
        assert edge.isVisible() and edge.is_collapsed()
        assert not window._dock_slot.isVisible()
        window.apply_dock_mode('hidden')
        assert edge.isVisible() and edge.is_collapsed()
        window.toggle_app_drawer()
        assert edge.isVisible() and window._dock_slot.isVisible()
        assert window._sidebar.width() == width
        assert preferences.get_dock_width() == 340
        assert window._stack.currentWidget() is module
        assert value.text() == 'unsaved module setting'
        assert preferences.get_dock_mode() == legacy_mode
        assert not preferences.get_easy_mode_dock()
    finally:
        window.close()
