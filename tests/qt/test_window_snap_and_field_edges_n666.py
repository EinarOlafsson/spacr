"""Window and panel gestures publish feedback after their geometry completes."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QDialog, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from spacr.qt.app import MainWindow
from spacr.qt.widgets import ambient, glass
from spacr.qt.widgets.collapsible_splitter import EDGE, CollapsibleSplitter
from spacr.qt.widgets.dock import Dock, DockEdge
from spacr.qt.widgets.foldable import make_foldable
from spacr.qt.widgets.height_grip import HeightGrip
from spacr.qt.widgets.section import Section


class _Shell(MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self._drag_from = None
        self.resize(320, 240)
        self.menuBar().addMenu("Actions")
        self.menuBar().installEventFilter(self)

    def _relay_the_menu_bar(self):
        pass


@pytest.fixture
def feedback(monkeypatch):
    calls = []

    def record(widget, edge=None, rect=None, strength=1.0):
        calls.append((widget, edge, QRect(widget.geometry()), rect, strength))

    monkeypatch.setattr(ambient, "field_ripple_for_widget", record, raising=False)
    return calls


@pytest.fixture
def shell(qtbot, feedback):
    window = _Shell()
    qtbot.addWidget(window)
    window.show()
    return window


def _mouse(kind, local, global_pos, button=Qt.LeftButton, buttons=Qt.LeftButton):
    return QMouseEvent(kind, QPointF(local), QPointF(global_pos), button,
                       buttons, Qt.NoModifier)


@pytest.mark.parametrize("edge", ["top", "left", "right", "bottom"])
def test_snaps_use_current_screen_and_report_the_same_edge(shell, feedback, edge, qtbot):
    desktop = shell.screen().availableGeometry()
    points = {"top": QPoint(desktop.center().x(), desktop.top()),
              "left": QPoint(desktop.left(), desktop.center().y()),
              "right": QPoint(desktop.right(), desktop.center().y()),
              "bottom": QPoint(desktop.center().x(), desktop.bottom())}
    assert shell._snap_to_screen_edge(points[edge])
    if edge == "top":
        assert shell.isMaximized()
        assert not shell.isFullScreen()
        assert not glass._edges_at(shell, QPoint(1, 1))
    else:
        expected = QRect(desktop)
        if edge == "left":
            expected.setWidth(desktop.width() // 2)
        elif edge == "right":
            expected.setLeft(desktop.left() + desktop.width() // 2)
        else:
            expected.setTop(desktop.top() + desktop.height() // 2)
        assert shell.geometry() == expected
        assert not shell.isFullScreen()
    qtbot.waitUntil(lambda: bool(feedback))
    assert len(feedback) == 1
    assert feedback[-1][0:2] == (shell, edge)
    assert feedback[-1][2] == shell.geometry()


def test_interior_release_moves_without_snapping_or_reseeding(shell, feedback):
    desktop = shell.screen().availableGeometry()
    shell.move(desktop.center() - QPoint(160, 120))
    before = QRect(shell.geometry())
    assert not shell._snap_to_screen_edge(desktop.center())
    assert shell.geometry() == before
    assert feedback[-1][0:2] == (shell, None)
    assert not shell._snap_to_screen_edge(QPoint(desktop.right() + 100, desktop.top()))
    assert len(feedback) == 1


def test_title_drag_requires_held_left_button_and_a_real_release(shell, feedback):
    bar = shell.menuBar()
    local = QPoint(bar.width() - 40, max(1, bar.height() // 2))
    assert bar.actionAt(local) is None
    origin = bar.mapToGlobal(local)
    assert shell.eventFilter(bar, _mouse(QEvent.MouseButtonPress, local, origin))
    before = shell.pos()
    assert not shell.eventFilter(bar, _mouse(QEvent.MouseMove, local,
                                            origin + QPoint(30, 20),
                                            Qt.NoButton, Qt.NoButton))
    assert shell.pos() == before
    assert shell._drag_from is None
    assert not feedback
    shell.eventFilter(bar, _mouse(QEvent.MouseButtonPress, local, origin))
    assert shell.eventFilter(bar, _mouse(QEvent.MouseMove, local,
                                         origin + QPoint(30, 20), Qt.NoButton))
    assert shell.pos() == before + QPoint(30, 20)
    screen = shell.screen().availableGeometry()
    assert shell.eventFilter(bar, _mouse(QEvent.MouseButtonRelease, local,
                                         QPoint(screen.left(), screen.center().y()),
                                         Qt.LeftButton, Qt.NoButton))
    assert shell._drag_from is None
    assert feedback[-1][1] == "left"


def test_wayland_title_press_remains_with_compositor(shell, monkeypatch):
    monkeypatch.setattr(glass.QApplication, "platformName", lambda: "wayland")
    bar = shell.menuBar()
    local = QPoint(bar.width() - 40, 10)
    assert not shell.eventFilter(bar, _mouse(QEvent.MouseButtonPress, local,
                                             bar.mapToGlobal(local)))
    assert shell._drag_from is None


@pytest.mark.parametrize("state", [Qt.WindowFullScreen, Qt.WindowMaximized])
def test_full_window_suppresses_hints_and_aborts_active_resize(qtbot, feedback, state):
    dialog = QDialog()
    qtbot.addWidget(dialog)
    dialog.resize(320, 240)
    dialog.show()
    watcher = glass._ResizeByEdge(dialog)
    point = QPoint(10, 100)
    origin = dialog.mapToGlobal(point)
    assert glass._edges_at(dialog, point) == Qt.LeftEdge
    assert watcher.eventFilter(dialog, _mouse(QEvent.MouseButtonPress, point, origin))
    dialog.setWindowState(state)
    before = QRect(dialog.geometry())
    watcher._hint.show_edges(Qt.LeftEdge)
    assert not watcher._hint.edges
    assert not watcher.eventFilter(dialog, _mouse(QEvent.MouseMove, point,
                                                   origin - QPoint(40, 0), Qt.NoButton))
    assert watcher._grab is None
    assert dialog.geometry() == before
    assert not watcher._hint.isVisible()
    assert not feedback


def test_expanded_hit_band_preserves_button_clicks_and_emits_after_resize(qtbot, feedback):
    dialog = QDialog()
    qtbot.addWidget(dialog)
    dialog.resize(320, 240)
    button = QPushButton("Action", dialog)
    button.setGeometry(0, 40, 80, 30)
    dialog.show()
    watcher = glass._ResizeByEdge(dialog)
    point = QPoint(10, 50)
    assert glass._edges_at(dialog, point) == Qt.LeftEdge
    assert not watcher.eventFilter(dialog, _mouse(QEvent.MouseButtonPress, point,
                                                   dialog.mapToGlobal(point)))
    clicked = []
    button.clicked.connect(lambda: clicked.append(True))
    qtbot.mouseClick(button, Qt.LeftButton, pos=QPoint(10, 10))
    assert clicked == [True]
    point = QPoint(10, 130)
    origin = dialog.mapToGlobal(point)
    assert watcher.eventFilter(dialog, _mouse(QEvent.MouseButtonPress, point, origin))
    watcher.eventFilter(dialog, _mouse(QEvent.MouseMove, point, origin - QPoint(30, 0), Qt.NoButton))
    assert not feedback
    watcher.eventFilter(dialog, _mouse(QEvent.MouseButtonRelease, point, origin - QPoint(30, 0),
                                       Qt.LeftButton, Qt.NoButton))
    assert feedback[-1][0] is dialog
    assert feedback[-1][2] == dialog.geometry()


def test_splitter_reports_completed_collapse_and_drag_only(qtbot, feedback):
    splitter = CollapsibleSplitter(Qt.Horizontal)
    qtbot.addWidget(splitter)
    splitter.resize(600, 260)
    splitter.add_pane(QWidget(), "Left", mode=EDGE, extent=200)
    splitter.add_pane(QWidget(), "Right", extent=300)
    splitter.show()
    assert splitter.set_collapsed("Left", True, by_user=True)
    assert feedback[-1][0] is splitter
    count = len(feedback)
    assert splitter.set_collapsed("Left", True, by_user=True)
    assert len(feedback) == count
    splitter.set_collapsed("Left", False, by_user=True)
    assert len(feedback) == count + 1
    assert feedback[-1][0] is splitter
    feedback.clear()
    handle = splitter.handle(1)
    local = handle.rect().center()
    origin = handle.mapToGlobal(local)
    handle.mousePressEvent(_mouse(QEvent.MouseButtonPress, local, origin))
    handle.mouseMoveEvent(_mouse(QEvent.MouseMove, local, origin + QPoint(30, 0), Qt.NoButton))
    assert not feedback
    handle.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, local, origin + QPoint(30, 0),
                                    Qt.LeftButton, Qt.NoButton))
    assert feedback[-1][0] is splitter


def test_header_fold_wave_uses_the_final_landing_edge(qtbot, feedback):
    splitter = CollapsibleSplitter(Qt.Horizontal)
    qtbot.addWidget(splitter)
    splitter.resize(600, 260)
    pane = QWidget()
    column = QVBoxLayout(pane)
    heading, body = QLabel("Pane"), QWidget()
    column.addWidget(heading)
    column.addWidget(body)
    folder = make_foldable(heading, body, name="Pane")
    splitter.add_pane(pane, "Pane", folder=folder)
    splitter.add_pane(QWidget(), "Rest")
    splitter.show()
    folder.toggle()
    assert feedback[-1][0] is splitter
    assert feedback[-1][1] == "right"
    assert feedback[-1][3] == pane.geometry()
    feedback.clear()
    folder.toggle()
    assert len(feedback) == 1
    assert feedback[-1][1] == "right"
    assert feedback[-1][3] == pane.geometry()


def test_dock_and_height_grips_publish_final_geometry(qtbot, feedback):
    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    dock = Dock([], parent=host)
    edge = DockEdge(dock, host)
    target = QWidget(host)
    target.setFixedHeight(100)
    grip = HeightGrip(target, 50, 300, host)
    for widget in (dock, edge, target, grip):
        layout.addWidget(widget)
    host.resize(500, 600)
    host.show()
    edge.show()
    edge.set_collapsed(True)
    qtbot.waitUntil(lambda: bool(feedback))
    assert feedback[-1][0] is edge
    feedback.clear()
    edge.set_collapsed(False)
    qtbot.waitUntil(lambda: bool(feedback))
    assert len(feedback) == 1
    assert feedback[-1][0] is dock
    assert feedback[-1][1] == "right"
    feedback.clear()
    point = edge.rect().center()
    origin = edge.mapToGlobal(point)
    edge.mousePressEvent(_mouse(QEvent.MouseButtonPress, point, origin))
    edge.mouseMoveEvent(_mouse(QEvent.MouseMove, point, origin + QPoint(30, 0), Qt.NoButton))
    assert not feedback
    edge.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, point, origin + QPoint(30, 0),
                                  Qt.LeftButton, Qt.NoButton))
    assert feedback[-1][0] is dock
    feedback.clear()
    point = grip.rect().center()
    origin = grip.mapToGlobal(point)
    grip.mousePressEvent(_mouse(QEvent.MouseButtonPress, point, origin))
    grip.mouseMoveEvent(_mouse(QEvent.MouseMove, point, origin + QPoint(0, 40), Qt.NoButton))
    assert not feedback
    grip.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, point, origin + QPoint(0, 40),
                                  Qt.LeftButton, Qt.NoButton))
    assert target.height() == 140
    assert feedback[-1][0] is target
    assert feedback[-1][2] == target.geometry()


def test_section_collapse_feedback_observes_hidden_body(qtbot, feedback):
    section = Section("Settings", expanded=True)
    qtbot.addWidget(section)
    section.show()
    feedback.clear()
    section._on_toggle(False)
    assert section._body.isHidden()
    qtbot.waitUntil(lambda: bool(feedback))
    assert feedback[-1][0] is section
    assert feedback[-1][1] == "bottom"
    assert feedback[-1][2] == section.geometry()
    feedback.clear()
    section._on_toggle(True)
    assert not section._body.isHidden()
    qtbot.waitUntil(lambda: bool(feedback))
    assert len(feedback) == 1
    assert feedback[-1][0] is section
    assert feedback[-1][1] == "bottom"


@pytest.mark.parametrize("state", [Qt.WindowFullScreen, Qt.WindowMaximized])
def test_native_full_window_state_reports_one_settled_top_ripple(
        shell, feedback, qtbot, state):
    shell.setWindowState(state)
    assert not feedback
    qtbot.waitUntil(lambda: bool(feedback))
    assert len(feedback) == 1
    assert feedback[0][0:2] == (shell, "top")
    assert feedback[0][2] == shell.geometry()
    shell.showNormal()
    shell.showMinimized()
    qtbot.wait(1)
    assert len(feedback) == 1


def test_fullscreen_control_reports_top_and_restore_does_not(shell, feedback, qtbot):
    assert shell.toggle_fullscreen()
    assert not feedback
    qtbot.waitUntil(lambda: bool(feedback))
    assert feedback[0][0:2] == (shell, "top")
    assert shell.toggle_fullscreen() is False
    qtbot.wait(1)
    assert len(feedback) == 1


@pytest.mark.parametrize("cancel", ["restore", "minimize", "delete"])
def test_top_feedback_is_cancelled_if_window_restores_or_dies_before_dispatch(
        shell, feedback, qtbot, cancel):
    shell.showMaximized()
    assert not feedback
    if cancel == "delete":
        from shiboken6 import delete as destroy

        destroy(shell)
    elif cancel == "minimize":
        shell.showMinimized()
    else:
        shell.showNormal()
    qtbot.wait(1)
    assert not feedback


@pytest.mark.parametrize("edge", ["left", "right", "bottom"])
def test_small_desktop_tiles_restore_the_original_minimum_on_next_drag(shell, feedback, monkeypatch, edge):
    class Desktop:
        def availableGeometry(self):
            return QRect(-1920, 40, 1920, 1080)

    monkeypatch.setattr(glass.QApplication, "screenAt", lambda _point: Desktop())
    shell.setMinimumSize(1200, 720)
    desktop = Desktop().availableGeometry()
    point = {"left": QPoint(desktop.left(), desktop.center().y()),
             "right": QPoint(desktop.right(), desktop.center().y()),
             "bottom": QPoint(desktop.center().x(), desktop.bottom())}[edge]
    assert shell._snap_to_screen_edge(point)
    assert shell.geometry().width() == (960 if edge != "bottom" else 1920)
    assert shell.geometry().height() == (540 if edge == "bottom" else 1080)
    assert shell._snap_minimum.width() == 1200
    assert shell._snap_minimum.height() == 720
    bar = shell.menuBar()
    local = QPoint(bar.width() - 40, 10)
    assert shell.eventFilter(bar, _mouse(QEvent.MouseButtonPress, local, bar.mapToGlobal(local)))
    assert shell.minimumWidth() == 1200
    assert shell.minimumHeight() == 720
    assert shell._snap_minimum is None


def test_popup_drag_feedback_requires_actual_movement(qtbot, feedback):
    dialog = QDialog()
    qtbot.addWidget(dialog)
    dialog.resize(320, 240)
    dialog.show()
    dragger = glass._DragByBackground(dialog)
    point = QPoint(150, 100)
    origin = dialog.mapToGlobal(point)
    dragger.eventFilter(dialog, _mouse(QEvent.MouseButtonPress, point, origin))
    dragger.eventFilter(dialog, _mouse(QEvent.MouseButtonRelease, point, origin,
                                      Qt.LeftButton, Qt.NoButton))
    assert not feedback
    dragger.eventFilter(dialog, _mouse(QEvent.MouseButtonPress, point, origin))
    dragger.eventFilter(dialog, _mouse(QEvent.MouseMove, point, origin + QPoint(40, 30), Qt.NoButton))
    assert not feedback
    dragger.eventFilter(dialog, _mouse(QEvent.MouseButtonRelease, point, origin + QPoint(40, 30),
                                      Qt.LeftButton, Qt.NoButton))
    assert feedback[-1][0] is dialog
    assert feedback[-1][2] == dialog.geometry()


def test_a_missing_screen_does_not_change_window_or_emit_feedback(shell, feedback, monkeypatch):
    monkeypatch.setattr(glass.QApplication, "screenAt", lambda _point: None)
    monkeypatch.setattr(shell, "screen", lambda: None)
    before = QRect(shell.geometry())
    assert not shell._snap_to_screen_edge(QPoint(0, 0))
    assert shell.geometry() == before
    assert not feedback


def test_clicks_on_resize_handles_do_not_emit_motion_feedback(qtbot, feedback):
    dialog = QDialog()
    qtbot.addWidget(dialog)
    dialog.resize(320, 240)
    dialog.show()
    watcher = glass._ResizeByEdge(dialog)
    point = QPoint(10, 130)
    origin = dialog.mapToGlobal(point)
    watcher.eventFilter(dialog, _mouse(QEvent.MouseButtonPress, point, origin))
    watcher.eventFilter(dialog, _mouse(QEvent.MouseButtonRelease, point, origin,
                                       Qt.LeftButton, Qt.NoButton))
    assert not feedback
    target = QWidget(dialog)
    target.setFixedHeight(100)
    grip = HeightGrip(target, 50, 300, dialog)
    grip.mousePressEvent(_mouse(QEvent.MouseButtonPress, point, origin))
    grip.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, point, origin,
                                  Qt.LeftButton, Qt.NoButton))
    assert not feedback


def test_fullscreen_and_maximize_restore_normal_minimum_after_a_tile(shell, feedback):
    shell.setMinimumSize(500, 500)
    desktop = shell.screen().availableGeometry()
    shell._snap_to_screen_edge(QPoint(desktop.left(), desktop.center().y()))
    assert shell._snap_minimum is not None
    bar = shell.menuBar()
    local = QPoint(bar.width() - 40, 10)
    assert shell.eventFilter(bar, _mouse(QEvent.MouseButtonDblClick, local,
                                        bar.mapToGlobal(local)))
    assert shell.isMaximized()
    assert shell._snap_minimum is None
    assert shell.minimumWidth() == 500
    shell.showNormal()
    shell._snap_to_screen_edge(QPoint(desktop.left(), desktop.center().y()))
    shell.toggle_fullscreen()
    assert shell._snap_minimum is None
    assert shell.minimumHeight() == 500
    assert shell.toggle_fullscreen() is False


def test_top_snap_from_a_tile_restores_minimum_before_maximising(shell, feedback):
    shell.setMinimumSize(500, 500)
    desktop = shell.screen().availableGeometry()
    shell._snap_to_screen_edge(QPoint(desktop.left(), desktop.center().y()))
    shell._snap_to_screen_edge(QPoint(desktop.center().x(), desktop.top()))
    assert shell.isMaximized()
    assert not shell.isFullScreen()
    assert shell.minimumWidth() == 500
    assert shell._snap_minimum is None


def test_repeated_side_snap_keeps_original_minimum_and_simple_click_does_not_snap(shell, feedback):
    shell.setMinimumSize(500, 500)
    desktop = shell.screen().availableGeometry()
    shell._snap_to_screen_edge(QPoint(desktop.left(), desktop.center().y()))
    shell._snap_to_screen_edge(QPoint(desktop.right(), desktop.center().y()))
    assert shell._snap_minimum.width() == 500
    feedback.clear()
    bar = shell.menuBar()
    local = QPoint(bar.width() - 40, 10)
    origin = bar.mapToGlobal(local)
    shell.eventFilter(bar, _mouse(QEvent.MouseButtonPress, local, origin))
    shell.eventFilter(bar, _mouse(QEvent.MouseButtonRelease, local, origin,
                                 Qt.LeftButton, Qt.NoButton))
    assert not feedback
    shell.showMaximized()
    assert shell.eventFilter(bar, _mouse(QEvent.MouseButtonPress, local, origin))
    assert not shell.isMaximized()


def test_middle_of_a_window_is_not_a_resize_target(qtbot, feedback):
    dialog = QDialog()
    qtbot.addWidget(dialog)
    dialog.resize(320, 240)
    dialog.show()
    watcher = glass._ResizeByEdge(dialog)
    point = QPoint(150, 100)
    assert not watcher.eventFilter(dialog, _mouse(QEvent.MouseButtonPress, point,
                                                   dialog.mapToGlobal(point)))
    assert not feedback


def test_scrolled_category_collapse_waits_for_its_layout(qtbot, feedback):
    from PySide6.QtWidgets import QScrollArea

    host = QScrollArea()
    qtbot.addWidget(host)
    section = Section("Settings", expanded=True)
    host.setWidget(section)
    host.setWidgetResizable(True)
    host.resize(400, 300)
    host.show()
    feedback.clear()
    section._on_toggle(False)
    assert section._body.isHidden()
    assert host.updatesEnabled()
    qtbot.waitUntil(lambda: bool(feedback))
    assert feedback[-1][0] is section


def test_existing_window_state_controls_work_without_a_snap(shell):
    bar = shell.menuBar()
    local = QPoint(bar.width() - 40, 10)
    event = _mouse(QEvent.MouseButtonDblClick, local, bar.mapToGlobal(local))
    assert shell.eventFilter(bar, event)
    assert shell.isMaximized()
    assert shell.eventFilter(bar, event)
    assert not shell.isMaximized()
    assert shell.toggle_fullscreen() is True
    assert shell.toggle_fullscreen() is False


def test_nested_category_publishes_feedback_on_the_completed_event_turn(qtbot, feedback):
    from PySide6.QtWidgets import QScrollArea
    from spacr.qt.widgets.section import SETTLING_DEPTH

    host = QScrollArea()
    qtbot.addWidget(host)
    section = Section("Nested", expanded=True)
    host.setWidget(section)
    host.show()
    qtbot.waitUntil(lambda: bool(feedback))
    feedback.clear()
    host.setProperty(SETTLING_DEPTH, 1)
    host.setUpdatesEnabled(False)
    section._on_toggle(False)
    assert not feedback
    assert host.property(SETTLING_DEPTH) == 1
    assert not host.updatesEnabled()
    host.setProperty(SETTLING_DEPTH, 0)
    host.setUpdatesEnabled(True)
    qtbot.waitUntil(lambda: bool(feedback))
    assert section._body.isHidden()
    assert feedback[-1][0] is section
