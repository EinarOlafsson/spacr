"""The rendered Navigator Dock action opens the module Dock independently."""
import pytest
from PySide6.QtCore import QPoint, QRect
from PySide6.QtTest import QSignalSpy

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile

pytestmark = pytest.mark.qt


def test_rendered_dock_link_opens_left_modules_without_reflowing_the_input(qtbot, private_ai_profile):
    from spacr.qt.app import MainWindow
    from tests.qt.test_739_navigator_protocols import _click_url

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    preferences.set_ambient_enabled(False)
    preferences._settings().setValue('prefs/navigator_dock_collapsed', True)
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1366, 768)
    window.show()
    page = window._startup
    split = page._navigator_aside_split
    split.set_collapsed('Widgets', True, by_user=False)
    qtbot.waitUntil(lambda: page.field.width() > 1000)
    assert window._dock_edge.is_collapsed()
    assert not window._dock_slot.isVisible()
    assert 'spacr://sidebar' in page.field.details.toHtml()
    anchor = QRect(page.path_edit.mapToGlobal(QPoint()), page.path_edit.size())
    aside_state = (QRect(page._navigator_aside.geometry()), split.mask())
    text = page.path_edit.text()
    result = page._last_result
    opened = QSignalSpy(page.module_requested)
    toggles = QSignalSpy(window._dock_edge.collapsedChanged)
    try:
        for _ in range(2):
            _click_url(qtbot, page.field.details, 'spacr://sidebar')
            qtbot.waitUntil(lambda: window._dock_slot.isVisible())
            assert not window._dock_edge.is_collapsed()
            assert window._sidebar.isVisible()
            assert window._dock_slot.width() >= window._sidebar.width() > 0
            assert split.is_collapsed('Widgets')
            assert (QRect(page._navigator_aside.geometry()), split.mask()) == aside_state
            assert QRect(page.path_edit.mapToGlobal(QPoint()), page.path_edit.size()) == anchor
            assert window._stack.geometry() == window.centralWidget().rect()
            assert page.path_edit.text() == text and page._last_result is result
            assert opened.count() == 0
        assert toggles.count() == 1 and toggles.at(0) == [False]
    finally:
        window.close()
