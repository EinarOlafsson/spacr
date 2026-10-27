"""Navigator commits searches and preserves stationary native controls."""
import pytest
from PySide6.QtCore import Qt, QPoint
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


@pytest.fixture
def navigator(qtbot, private_ai_profile):
    from spacr.qt.widgets.home import _EasyHomePage
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_ambient_enabled(True)
    preferences.set_reduce_motion(False)
    page = _EasyHomePage()
    page.resize(1080, 760)
    qtbot.addWidget(page)
    page.show()
    yield page
    page.close()


def choices(page, count=10):
    keys = ['mask', 'make_masks', 'measure', 'db_browser', 'graph_builder',
            'gate_editor', 'regression', 'annotate', 'convert', 'umap'][:count]
    actions = [(key, key, 'Explanation for ' + key, '/local/project') for key in keys]
    page._requested = page.path_edit.text()
    page._show_result(page._input_generation,
        dict(path='/local/project', message='', actions=actions, context={}))
    page.field._finish_growth()
    return page.field.cards


def test_typing_never_discovers_until_enter(navigator, qtbot, monkeypatch):
    from spacr.qt.widgets import home
    calls = []
    def inspect(value):
        calls.append(value)
        return dict(path=value, message='', actions=[], context={})
    monkeypatch.setattr(home, '_easy_inspect_path', inspect)
    navigator.path_edit.setFocus()
    qtbot.keyClicks(navigator.path_edit, '/local/input-images')
    qtbot.wait(750)
    assert not calls and not navigator.field.cards and not navigator._debounce.isActive()
    qtbot.keyClick(navigator.path_edit, Qt.Key_Return)
    qtbot.waitUntil(lambda: bool(calls), timeout=5000)
    assert calls == ['/local/input-images']


def test_name_only_cards_describe_in_stationary_lower_panel(navigator, qtbot):
    before = (navigator.path_edit.geometry(), navigator.field.details.geometry())
    assert abs(navigator.path_edit.geometry().center().y() - navigator.field.rect().center().y()) <= 1
    cards = choices(navigator)
    assert (navigator.path_edit.geometry(), navigator.field.details.geometry()) == before
    assert navigator.field.details.y() >= navigator.field.height() // 2
    for card in cards:
        assert not card.description.isVisible()
        assert not card.toolTip()
        assert card.title.text()
    qtbot.mouseMove(cards[1], QPoint(cards[1].width() // 2, cards[1].height() // 2))
    qtbot.waitUntil(lambda: 'make_masks' in navigator.field.details.toPlainText())
    assert (navigator.path_edit.geometry(), navigator.field.details.geometry()) == before


def test_choices_scroll_as_one_row_with_soft_edge_masks(navigator):
    cards = choices(navigator)
    field = navigator.field
    before = (field.panel.geometry(), field.details.geometry())
    assert len({card._final_rect.y() for card in cards}) == 1
    assert field._scrollbar.maximum() > 0
    assert not cards[-1].isVisible()
    field._scrollbar.setValue(field._scrollbar.maximum() // 2)
    shown = [card for card in cards if card.isVisible()]
    assert shown and all(not card.mask().isEmpty() for card in shown)
    assert all(card._content_opacity.opacityMask().gradient() is not None for card in shown)
    field._scrollbar.setValue(field._scrollbar.maximum())
    assert cards[-1].isVisible() and not cards[0].isVisible()
    assert (field.panel.geometry(), field.details.geometry()) == before


def test_network_preserves_fixed_controls_and_fades_departing_cards(navigator, qtbot):
    cards = choices(navigator, 3)
    field = navigator.field
    anchor = (field.panel.geometry(), field.details.geometry())
    selected, departing = cards[:2]
    qtbot.mouseClick(selected, Qt.LeftButton)
    workflow = navigator._workflow
    assert workflow is not None and workflow.details is field.details
    assert navigator.path_edit.isVisible()
    qtbot.waitUntil(lambda: .05 < workflow._progress < .9, timeout=2000)
    assert departing.isVisible() and 0 < departing._content_opacity.opacity() < 1
    assert not departing.isEnabled()
    assert (field.panel.geometry(), field.details.geometry()) == anchor
    qtbot.waitUntil(lambda: workflow._progress == 1, timeout=4000)
    assert not departing.isVisible()
    navigator._show_suggestions()
    assert departing.isVisible() and departing.isEnabled()
    assert departing._content_opacity.opacity() == 1
    assert (field.panel.geometry(), field.details.geometry()) == anchor


@pytest.mark.parametrize('width,height', [(520, 460), (1080, 760)])
def test_fixed_panels_do_not_overlap_choice_row(navigator, qtbot, width, height):
    navigator.resize(width, height)
    qtbot.wait(20)
    choices(navigator)
    field = navigator.field
    assert abs(field.panel.geometry().center().y() - field.rect().center().y()) <= 1
    assert not field.panel.geometry().intersects(field._row_rect)
    assert not field.details.geometry().intersects(field._row_rect)
    assert field.rect().contains(field.details.geometry())
    assert field.rect().contains(field.panel.geometry())


def test_network_fades_search_and_restores_same_pixels_and_position(navigator, qtbot):
    from PySide6.QtCore import QPoint
    choices(navigator, 3)
    field = navigator.field
    search = field.panel.mapTo(navigator, QPoint())
    point = search + QPoint(50, 10)
    before = navigator.grab().toImage().pixelColor(point).getRgb()
    assert before[:3] == (25, 25, 25)
    qtbot.mouseClick(field.cards[0], Qt.LeftButton)
    navigator._workflow._transition.setCurrentTime(500)
    qtbot.wait(30)
    assert not field.panel.isVisible()
    assert field.panel.mapTo(navigator, QPoint()) == search
    navigator._show_suggestions()
    qtbot.wait(30)
    assert field.panel.isVisible() and field.panel.mapTo(navigator, QPoint()) == search
    assert navigator.grab().toImage().pixelColor(point).getRgb() == before


def test_mouse_wheel_scrolls_only_the_choice_row(navigator, qtbot):
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication
    choices(navigator)
    field = navigator.field
    anchor = (field.panel.geometry(), field.details.geometry())
    point = field._row_rect.center()
    event = QWheelEvent(QPointF(point), QPointF(field.mapToGlobal(point)),
                        QPoint(), QPoint(0, -120), Qt.NoButton, Qt.NoModifier,
                        Qt.NoScrollPhase, False)
    QApplication.sendEvent(field, event)
    assert field._scrollbar.value() > 0 and event.isAccepted()
    assert (field.panel.geometry(), field.details.geometry()) == anchor
