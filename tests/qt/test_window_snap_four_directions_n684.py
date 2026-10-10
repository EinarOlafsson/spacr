"""N684: the frameless window snaps left, right, up and down, and full screen
stays a separate state."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QAction, QMouseEvent
from PySide6.QtWidgets import QMainWindow

from spacr.qt.app import MainWindow
from spacr.qt.widgets import ambient, glass


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
        calls.append((widget, edge, QRect(widget.geometry())))

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


def _expected(desktop, edge):
    target = QRect(desktop)
    if edge == "left":
        target.setWidth(desktop.width() // 2)
    elif edge == "right":
        target.setLeft(desktop.left() + desktop.width() // 2)
    elif edge == "down":
        target.setTop(desktop.top() + desktop.height() // 2)
    return target


@pytest.mark.parametrize("edge,ripple", [("left", "left"), ("right", "right"),
                                         ("up", "top"), ("down", "bottom")])
def test_each_direction_fills_its_part_of_the_work_area_with_one_ripple(
        shell, feedback, qtbot, edge, ripple):
    desktop = shell.screen().availableGeometry()
    assert shell.snap_window(edge) is True
    assert not shell.isFullScreen()
    if edge == "up":
        assert shell.isMaximized()
    else:
        assert not shell.isMaximized()
        assert shell.geometry() == _expected(desktop, edge)
    qtbot.waitUntil(lambda: bool(feedback))
    qtbot.wait(5)
    assert len(feedback) == 1
    assert feedback[0][0:2] == (shell, ripple)


def test_an_unknown_direction_changes_nothing(shell, feedback):
    before = QRect(shell.geometry())
    assert shell.snap_window("sideways") is False
    assert shell.geometry() == before
    assert not feedback


def test_up_when_already_maximised_still_ripples_once(shell, feedback, qtbot):
    shell.showMaximized()
    qtbot.waitUntil(lambda: bool(feedback))
    feedback.clear()
    assert shell.snap_window("up")
    qtbot.wait(5)
    assert len(feedback) == 1
    assert feedback[0][1] == "top"


def test_window_menu_carries_the_four_snaps(qtbot):
    window = _Shell()
    qtbot.addWidget(window)
    menu = window._build_window_menu(window.menuBar())
    names = {a.objectName(): a for a in menu.actions() if isinstance(a, QAction)}
    for name, edge in (("SnapLeftAction", "left"), ("SnapRightAction", "right"),
                       ("SnapUpAction", "up"), ("SnapDownAction", "down")):
        assert name in names
        assert names[name] is window._act_snaps[edge]
        assert names[name].text()
    assert window._act_fullscreen in menu.actions()
    assert window._act_fullscreen not in window._act_snaps.values()


def test_window_menu_snap_action_drives_snap_window(qtbot, feedback):
    window = _Shell()
    qtbot.addWidget(window)
    window._build_window_menu(window.menuBar())
    window.show()
    desktop = window.screen().availableGeometry()
    window._act_snaps["right"].trigger()
    assert window.geometry() == _expected(desktop, "right")


def test_full_screen_is_independent_of_the_up_snap(shell, feedback, qtbot):
    assert shell.snap_window("up")
    assert shell.isMaximized() and not shell.isFullScreen()
    assert shell.toggle_fullscreen() is True
    assert shell.isFullScreen()
    assert shell.toggle_fullscreen() is False
    assert shell.isMaximized()
    assert not shell.isFullScreen()


def test_leaving_full_screen_returns_to_a_tile(shell, feedback):
    desktop = shell.screen().availableGeometry()
    assert shell.snap_window("left")
    tile = QRect(shell.geometry())
    assert shell.toggle_fullscreen() is True
    assert shell.toggle_fullscreen() is False
    assert not shell.isMaximized()
    assert shell.geometry() == tile == _expected(desktop, "left")


def test_up_snap_leaves_full_screen_first(shell, feedback):
    shell.toggle_fullscreen()
    assert shell.snap_window("up")
    assert shell.isMaximized()
    assert not shell.isFullScreen()


@pytest.mark.parametrize("edge", ["left", "up", "down"])
def test_dragging_a_snapped_window_off_restores_its_size(shell, feedback, edge):
    desktop = shell.screen().availableGeometry()
    shell.move(desktop.topLeft() + QPoint(40, 40))
    size = shell.size()
    assert shell.snap_window(edge)
    assert shell._snap_restore.size() == size
    bar = shell.menuBar()
    local = QPoint(bar.width() - 40, max(1, bar.height() // 2))
    origin = bar.mapToGlobal(local)
    fraction = local.x() / max(1, shell.width())
    assert shell.eventFilter(bar, _mouse(QEvent.MouseButtonPress, local, origin))
    assert not shell.isMaximized()
    assert shell.size() == size
    assert shell._snap_restore is None
    assert getattr(shell, "_snap_minimum", None) is None
    assert abs(shell.frameGeometry().left()
               - (origin.x() - int(size.width() * fraction))) <= 1


def test_maximise_toggle_from_an_up_snap_restores_the_original_geometry(shell, feedback):
    desktop = shell.screen().availableGeometry()
    shell.move(desktop.topLeft() + QPoint(30, 30))
    before = QRect(shell.geometry())
    assert shell.snap_window("up")
    assert shell._toggle_maximised() is False
    assert shell.geometry() == before


def test_second_snap_keeps_the_first_restore_geometry(shell, feedback):
    size = shell.size()
    shell.snap_window("left")
    shell.snap_window("right")
    shell.snap_window("up")
    assert shell._snap_restore.size() == size


def test_a_drag_to_the_top_maximises_and_never_goes_full_screen(shell, feedback, qtbot):
    bar = shell.menuBar()
    local = QPoint(bar.width() - 40, max(1, bar.height() // 2))
    origin = bar.mapToGlobal(local)
    desktop = shell.screen().availableGeometry()
    shell.eventFilter(bar, _mouse(QEvent.MouseButtonPress, local, origin))
    shell.eventFilter(bar, _mouse(QEvent.MouseMove, local, origin + QPoint(10, 10),
                                  Qt.NoButton))
    top = QPoint(desktop.center().x(), desktop.top())
    assert shell.eventFilter(bar, _mouse(QEvent.MouseButtonRelease, local, top,
                                         Qt.LeftButton, Qt.NoButton))
    assert shell.isMaximized()
    assert not shell.isFullScreen()
    qtbot.waitUntil(lambda: bool(feedback))
    qtbot.wait(5)
    assert [f[1] for f in feedback] == ["top"]


def test_interior_release_does_not_snap(shell, feedback):
    desktop = shell.screen().availableGeometry()
    shell.move(desktop.center() - QPoint(160, 120))
    before = QRect(shell.geometry())
    assert not shell._snap_to_screen_edge(desktop.center())
    assert shell.geometry() == before
    assert not shell.isMaximized()
    assert getattr(shell, "_snap_restore", None) is None


def test_drag_snaps_on_the_screen_under_the_pointer(shell, feedback, monkeypatch):
    class Other:
        def availableGeometry(self):
            return QRect(-1600, 30, 1600, 870)

    monkeypatch.setattr(glass.QApplication, "screenAt", lambda _point: Other())
    desktop = Other().availableGeometry()
    assert shell._snap_to_screen_edge(QPoint(desktop.right(), desktop.center().y()))
    assert shell.geometry() == _expected(desktop, "right")


def test_wayland_drag_stays_with_the_compositor_and_halves_do_not_claim_position(
        shell, feedback, monkeypatch):
    monkeypatch.setattr(glass.QApplication, "platformName", lambda: "wayland")
    bar = shell.menuBar()
    local = QPoint(bar.width() - 40, 10)
    assert not shell.eventFilter(bar, _mouse(QEvent.MouseButtonPress, local,
                                             bar.mapToGlobal(local)))
    assert shell._drag_from is None
    assert shell.snap_window("left") is False
    assert shell.snap_window("up") is True
    assert shell.isMaximized()


def test_no_screen_means_no_snap(shell, feedback, monkeypatch):
    monkeypatch.setattr(shell, "screen", lambda: None)
    before = QRect(shell.geometry())
    assert shell.snap_window("left") is False
    assert shell.geometry() == before
    assert not feedback
