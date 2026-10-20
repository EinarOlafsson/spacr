"""N684: snapping and full screen at their edges.

An empty work area refuses a snap, an up snap onto another screen first
moves the window there, a half snap from a maximised window leaves the
maximised state, and leaving a full screen that was not entered through
``toggle_fullscreen`` falls back to an ordinary window without a geometry.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QMainWindow

from spacr.qt.app import MainWindow
from spacr.qt.widgets import ambient


class _Shell(MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self._drag_from = None
        self.resize(320, 240)
        self.menuBar().addMenu("Actions")

    def _relay_the_menu_bar(self):
        pass


class _Screen:
    def __init__(self, rect):
        self.rect = QRect(rect)

    def availableGeometry(self):
        return QRect(self.rect)


@pytest.fixture
def feedback(monkeypatch):
    calls = []
    monkeypatch.setattr(ambient, "field_ripple_for_widget",
                        lambda widget, edge=None, rect=None, strength=1.0:
                        calls.append(edge), raising=False)
    return calls


@pytest.fixture
def shell(qtbot, feedback):
    window = _Shell()
    qtbot.addWidget(window)
    window.show()
    return window


@pytest.mark.parametrize("edge", ["left", "up"])
def test_an_empty_work_area_refuses_the_snap_and_remembers_nothing(shell, feedback, edge):
    before = QRect(shell.geometry())
    assert shell.snap_window(edge, screen=_Screen(QRect())) is False
    assert shell.geometry() == before
    assert getattr(shell, "_snap_restore", None) is None
    assert not shell.isMaximized()
    assert feedback == []


def test_up_snap_on_another_screen_moves_there_before_maximising(shell, monkeypatch):
    other = _Screen(QRect(2000, 40, 1600, 900))
    moves = []
    original = shell.move
    monkeypatch.setattr(shell, "move", lambda *args: (moves.append(QPoint(*args)),
                                                      original(*args)))
    assert shell.snap_window("up", screen=other) is True
    assert moves == [QPoint(2000, 40)]
    assert shell.isMaximized()


def test_up_snap_on_its_own_screen_does_not_move_first(shell, monkeypatch):
    moves = []
    monkeypatch.setattr(shell, "move", lambda *args: moves.append(args))
    assert shell.snap_window("up", screen=shell.screen()) is True
    assert moves == [] and shell.isMaximized()


def test_a_half_snap_from_a_maximised_window_leaves_maximised(shell, feedback, qtbot):
    shell.showMaximized()
    qtbot.waitUntil(shell.isMaximized)
    desktop = shell.screen().availableGeometry()
    assert shell.snap_window("left") is True
    assert not shell.isMaximized()
    expected = QRect(desktop)
    expected.setWidth(desktop.width() // 2)
    assert shell.geometry() == expected
    assert "left" in feedback


def test_leaving_a_full_screen_entered_elsewhere_returns_to_a_plain_window(shell, qtbot):
    shell.showFullScreen()
    qtbot.waitUntil(shell.isFullScreen)
    assert getattr(shell, "_before_fullscreen", None) is None
    assert shell.toggle_fullscreen() is False
    assert not shell.isFullScreen() and not shell.isMaximized()
    assert shell._before_fullscreen is None
    assert getattr(shell, "_snap_minimum", None) is None


def test_leaving_full_screen_ignores_an_invalid_remembered_geometry(shell, qtbot):
    shell.showFullScreen()
    qtbot.waitUntil(shell.isFullScreen)
    shell._before_fullscreen = (False, QRect(), None)
    calls = []
    shell.setGeometry = lambda *args: calls.append(args)
    assert shell.toggle_fullscreen() is False
    assert calls == []


@pytest.mark.parametrize("restore", [None, QRect()])
def test_leaving_a_snap_without_a_usable_restore_keeps_the_size(shell, restore):
    shell.resize(300, 200)
    shell._snap_restore = restore
    shell._snap_minimum = None
    size = shell.size()
    shell._leave_snap(place=True)
    assert shell.size() == size
    assert shell._snap_restore is None


def test_double_clicking_the_title_bar_maximises_and_restores(shell, qtbot):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    bar = shell.menuBar()
    local = QPointF(bar.width() - 30, 5)
    event = QMouseEvent(QEvent.MouseButtonDblClick, local, bar.mapToGlobal(local),
                        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    assert shell.eventFilter(bar, event) is True
    qtbot.waitUntil(shell.isMaximized)
    assert shell.eventFilter(bar, event) is True
    qtbot.waitUntil(lambda: not shell.isMaximized())


@pytest.mark.parametrize("where,edge", [("left", "left"), ("bottom", "down")])
def test_a_drag_released_on_a_side_or_the_bottom_tiles_there(shell, feedback, where,
                                                             edge, monkeypatch):
    desktop = shell.screen().availableGeometry()
    seen = []
    monkeypatch.setattr(shell, "snap_window",
                        lambda direction, screen=None: seen.append(direction) or True)
    point = (QPoint(desktop.left() + 2, desktop.center().y()) if where == "left"
             else QPoint(desktop.center().x(), desktop.bottom() - 2))
    assert shell._snap_to_screen_edge(point) is True
    assert seen == [edge]


def test_maximise_toggle_restores_a_plain_window_and_the_snap_minimum(shell, qtbot):
    from PySide6.QtCore import QSize

    shell._snap_restore = None
    shell.showMaximized()
    qtbot.waitUntil(shell.isMaximized)
    assert shell._toggle_maximised() is False
    qtbot.waitUntil(lambda: not shell.isMaximized())
    shell.setMinimumSize(10, 10)
    shell._snap_minimum = QSize(200, 150)
    assert shell._toggle_maximised() is True
    assert shell.minimumSize() == QSize(200, 150)
    assert shell._snap_minimum is None
    assert shell._toggle_maximised() is False
    shell.setMinimumSize(0, 0)
    assert shell._toggle_maximised() is True
    assert shell.minimumSize() == QSize(0, 0)
