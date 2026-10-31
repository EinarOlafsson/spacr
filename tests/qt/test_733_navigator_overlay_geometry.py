"""Navigator side controls overlay a stationary, interactive central viewport."""
import pytest
from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QApplication, QWidget

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


def _rects(widgets):
    return tuple(QRect(widget.mapToGlobal(QPoint()), widget.size()) for widget in widgets)


class _GeometryTrace(QObject):
    def __init__(self, widgets, parent):
        super().__init__(parent)
        self.widgets = widgets
        self.frames = []
        for widget in widgets:
            widget.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Move, QEvent.Resize):
            self.frames.append(_rects(self.widgets))
        return False


@pytest.fixture
def make_window(qtbot, private_ai_profile):
    from spacr.qt.app import MainWindow

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    preferences.set_ambient_enabled(False)
    windows = []

    def build(collapsed=True):
        preferences._settings().setValue('prefs/navigator_dock_collapsed', collapsed)
        window = MainWindow(initial_app='__home__')
        qtbot.addWidget(window)
        windows.append(window)
        window._tour_timer.stop()
        window._consent_timer.stop()
        window.resize(1280, 900)
        window.show()
        qtbot.waitUntil(lambda: window._startup.field.width() > 1000)
        return window

    yield build
    for window in windows:
        window.close()


def _choices(page):
    page.path_edit.setText('/local/project')
    page._requested = page.path_edit.text()
    page._show_result(page._input_generation, dict(
        path=page.path_edit.text(), message='', context={},
        actions=[(key, key, 'Explanation', '/local/project')
                 for key in ('mask', 'measure', 'graph_builder')]))
    page.field._finish_growth()
    return page.field.cards


def _composer_rect(page):
    field = page.field
    width = min(720, max(180, field.width() - 48))
    split = page._navigator_aside_split
    if not split.is_collapsed('Widgets'):
        edge = field.mapFromGlobal(page._navigator_aside.mapToGlobal(QPoint())).x()
        width = min(width, max(180, 2 * (edge - 16) - field.width()))
    top = max(16, (field.height() - field.panel.height()) // 2)
    return QRect(field.mapToGlobal(QPoint((field.width() - width) // 2, top)),
                 QSize(width, field.panel.height()))


def _composer_anchor(rect):
    return rect.x() * 2 + rect.width(), rect.y(), rect.height()


def _toggle_sides(window, qtbot, tracked, trace):
    page = window._startup
    split = page._navigator_aside_split
    baseline = _rects(tracked)
    panel_index = tracked.index(page.field.panel)
    assert baseline[panel_index] == _composer_rect(page)
    for control in (window._dock_edge, split.handle(1)):
        for _ in range(2):
            before = _rects(tracked)[panel_index]
            qtbot.mouseClick(control, Qt.LeftButton)
            qtbot.wait(40)
            expected = list(baseline)
            expected[panel_index] = _composer_rect(page)
            assert _rects(tracked) == tuple(expected)
            for frame in trace.frames:
                assert frame[:panel_index] == baseline[:panel_index]
                assert frame[panel_index + 1:] == baseline[panel_index + 1:]
                assert _composer_anchor(frame[panel_index]) == _composer_anchor(baseline[panel_index])
                assert frame[panel_index].width() in {before.width(), expected[panel_index].width()}
            if not split.is_collapsed('Widgets'):
                aside_left = page._navigator_aside.mapToGlobal(QPoint()).x()
                assert expected[panel_index].right() + 1 <= aside_left - 16
            trace.frames.clear()
            assert window._stack.geometry() == window.centralWidget().rect()
            if page.path_edit.isVisible():
                point = page.path_edit.mapToGlobal(page.path_edit.rect().center())
                target = QApplication.widgetAt(point)
                assert target is page.path_edit or page.path_edit.isAncestorOf(target)


@pytest.mark.parametrize('saved_closed', [False, True])
def test_both_native_side_controls_leave_every_central_frame_stationary(
        qtbot, private_ai_profile, make_window, saved_closed):
    window = make_window(saved_closed)
    page = window._startup
    assert window._dock_edge.is_collapsed() is saved_closed
    cards = _choices(page)
    qtbot.wait(30)
    tracked = (page.field, page.field.panel, page.field.details, *cards)
    trace = _GeometryTrace(tracked, window)
    _toggle_sides(window, qtbot, tracked, trace)
    split = page._navigator_aside_split
    assert not split.mask().contains(split.mapFromGlobal(
        page.path_edit.mapToGlobal(page.path_edit.rect().center())))
    assert split.mask().contains(split.handle(1).geometry().center())
    assert window._dock_edge.is_collapsed() is saved_closed
    preferences, _, _ = private_ai_profile
    assert preferences._settings().value('prefs/navigator_dock_collapsed', type=bool) is saved_closed


def test_network_and_resize_keep_overlays_local_and_restore_ordinary_layout(qtbot, make_window):
    window = make_window()
    page = window._startup
    card = _choices(page)[0]
    qtbot.mouseClick(card, Qt.LeftButton)
    workflow = page._workflow
    assert workflow is not None
    qtbot.wait(40)
    for width, height in ((1280, 900), (1540, 1020)):
        window.resize(width, height)
        qtbot.wait(40)
        assert workflow.geometry() == QRect(
            page._suggestions_scroll.mapTo(page, QPoint()), page._suggestions_scroll.size())
        tracked = (page.field, workflow, workflow.central, page.field.panel, page.field.details)
        trace = _GeometryTrace(tracked, window)
        _toggle_sides(window, qtbot, tracked, trace)
        assert all(not choice.isVisible() for choice in page.field.cards
                   if choice is not workflow.central)
        for widget in tracked:
            widget.removeEventFilter(trace)
    ordinary = QWidget(window)
    window._stack.addWidget(ordinary)
    window._stack.setCurrentWidget(ordinary)
    qtbot.wait(30)
    row = window.centralWidget().layout()
    assert row.indexOf(window._dock_slot) == 0 and row.indexOf(window._dock_edge) == 1
    assert row.indexOf(window._stack) == 2
    window._dock_edge.set_collapsed(False)
    qtbot.wait(30)
    assert window._stack.x() >= window._dock_slot.width() + window._dock_edge.width()
    window._stack.setCurrentWidget(page)
    qtbot.wait(30)
    assert row.indexOf(window._dock_slot) == row.indexOf(window._dock_edge) == -1
    assert window._stack.geometry() == window.centralWidget().rect()


def test_passive_side_toggles_publish_no_field_feedback_but_ordinary_dock_does(
        qtbot, monkeypatch, make_window):
    from spacr.qt.widgets import ambient

    window = make_window()
    page = window._startup
    notes = []
    monkeypatch.setattr(ambient, '_note_container_ripple', lambda: notes.append(True))
    for control in (window._dock_edge, page._navigator_aside_split.handle(1)):
        qtbot.mouseClick(control, Qt.LeftButton)
        qtbot.wait(30)
    assert notes == []
    assert all(widget.property('spacrAmbientPassiveOverlay') is True for widget in (
        window._dock_slot, window._dock_edge, page._navigator_aside_split))
    ordinary = QWidget(window)
    window._stack.addWidget(ordinary)
    window._stack.setCurrentWidget(ordinary)
    qtbot.wait(30)
    notes.clear()
    qtbot.mouseClick(window._dock_edge, Qt.LeftButton)
    qtbot.waitUntil(lambda: bool(notes), timeout=1000)
