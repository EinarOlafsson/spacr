"""The lower description grows upward while search and choices keep their anchors."""
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from tests.qt.test_733_navigator_fixed_layout import choices
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


@pytest.fixture
def page(qtbot, private_ai_profile):
    from spacr.qt.widgets.home import _EasyHomePage
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_font_scale(1.0)
    preferences.set_reduce_motion(True)
    value = _EasyHomePage()
    qtbot.addWidget(value)
    value.resize(1100, 1000)
    value.show()
    qtbot.wait(10)
    yield value
    value.close()


def test_upward_drag_keeps_bottom_search_and_cards_fixed(page, qtbot):
    field = page.field
    cards = choices(page, 3)
    grip = field._details_grip
    before = (field.details.geometry(), field.panel.geometry(), [card.geometry() for card in cards])
    start = grip.rect().center()
    origin = grip.mapToGlobal(start)
    qtbot.mousePress(grip, Qt.LeftButton, pos=start)
    destination = origin - QPoint(0, 55)
    move = QMouseEvent(QEvent.MouseMove, QPointF(grip.mapFromGlobal(destination)),
                       QPointF(destination), Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(grip, move)
    qtbot.mouseRelease(grip, Qt.LeftButton, pos=grip.rect().center())
    assert field.details.height() == before[0].height() + 55
    assert field.details.geometry().bottom() == before[0].bottom()
    assert field.details.y() == before[0].y() - 55
    assert field.panel.geometry() == before[1]
    assert [card.geometry() for card in cards] == before[2]
    assert grip.accessibleName() and grip.focusPolicy() == Qt.StrongFocus


def test_keyboard_bounds_saved_choice_and_reset(page, qtbot, private_ai_profile):
    from spacr.qt.widgets.home import _EasyHomePage
    preferences, _, _ = private_ai_profile
    field, grip = page.field, page.field._details_grip
    initial = field.details.height()
    bottom, search = field.details.geometry().bottom(), field.panel.geometry()
    grip.setFocus()
    qtbot.keyClick(grip, Qt.Key_Up)
    assert field.details.height() == initial + 12
    qtbot.keyClick(grip, Qt.Key_Down)
    assert field.details.height() == initial
    qtbot.keyClick(grip, Qt.Key_End)
    assert field.details.height() == grip.bounds()[1]
    assert field.details.y() >= field.panel.geometry().bottom() + 16
    assert field.details.geometry().bottom() == bottom and field.panel.geometry() == search
    qtbot.keyClick(grip, Qt.Key_Home)
    assert field.details.height() == grip.bounds()[0]
    qtbot.keyClick(grip, Qt.Key_PageUp)
    desired = field.details.height()
    assert preferences._settings().contains('home/navigator_details_height')
    other = _EasyHomePage()
    qtbot.addWidget(other)
    other.resize(page.size())
    other.show()
    qtbot.wait(10)
    assert other.field.details.height() == desired
    page.resize(640, 460)
    qtbot.wait(10)
    assert field.details.geometry().bottom() == field.height() - 17
    assert field.details.y() >= field.panel.geometry().bottom() + 16
    page.resize(1100, 1000)
    qtbot.wait(10)
    assert field.details.height() == desired
    grip.reset()
    assert field.details.height() == initial
    assert not preferences._settings().contains('home/navigator_details_height')
    other.close()


def test_workflow_carries_accessible_grip_and_returns_it(page, qtbot):
    field = page.field
    card = choices(page, 3)[0]
    page._select_suggestion(card._key, card._path)
    assert page._workflow is not None
    grip = field._details_grip
    assert grip.parentWidget() is field.details
    assert field.details.parentWidget() is page._workflow
    previous_bottom = field.details.geometry().bottom()
    qtbot.keyClick(grip, Qt.Key_Up)
    assert field.details.geometry().bottom() == previous_bottom
    page._show_suggestions()
    assert field.details.parentWidget() is field and grip.isVisible()
