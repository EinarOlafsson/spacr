"""A frameless window resizes from the edge that was grabbed, and only while held.

Pinned here:

* a maximised window, or a point outside it, has no grab band;
* selectable text, a widget holding a selection, and a widget that says it
  owns its own gestures keep the press (the background drag stays out);
* a resize drag moves the left/top edge while the right/bottom stays put,
  and the right/bottom edge while the left/top stays put;
* a move that arrives after the button was let go (the release was lost)
  ends the resize or drag instead of continuing it;
* a jiggle smaller than the drag distance does not move the window;
* on Wayland the compositor is asked to resize or move, and the window does
  not move itself on top of it;
* pressing a child that sits on the window's edge starts a resize, because
  the app-wide filter forwards that press to the window's resizer.
"""
from __future__ import annotations

import types

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout, QWidget

from spacr.qt.widgets import glass


@pytest.fixture(autouse=True)
def _no_installer_left_behind():
    yield
    glass.uninstall_glass_everywhere()


class _HandledDialog(QDialog):
    """A dialog whose native handle is a recorder, so drags can be observed."""

    def __init__(self, move_accepted=False):
        super().__init__()
        self.asked_resize = []
        self.asked_move = []
        self.handle = types.SimpleNamespace(
            startSystemResize=lambda edges: self.asked_resize.append(edges) or True,
            startSystemMove=lambda: self.asked_move.append(True) or move_accepted,
        )

    def windowHandle(self):  # noqa: N802 - Qt naming
        return self.handle


@pytest.fixture
def dialog(qtbot):
    dlg = _HandledDialog()
    qtbot.addWidget(dlg)
    QVBoxLayout(dlg)
    dlg.setGeometry(100, 100, 320, 240)
    return dlg


def _press(pos, glob=(500, 500)):
    return QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(*pos),
                       QPointF(*glob), Qt.LeftButton, Qt.LeftButton,
                       Qt.NoModifier)


def _move(pos, glob, buttons=Qt.LeftButton):
    return QMouseEvent(QEvent.Type.MouseMove, QPointF(*pos), QPointF(*glob),
                       Qt.NoButton, buttons, Qt.NoModifier)


def _on_wayland(monkeypatch):
    monkeypatch.setattr(glass.QApplication, "platformName",
                        staticmethod(lambda: "wayland"))


def _off_wayland(monkeypatch):
    monkeypatch.setattr(glass.QApplication, "platformName",
                        staticmethod(lambda: "xcb"))


# ---------------------------------------------------------------------------
# Where the grab band is not
# ---------------------------------------------------------------------------

def test_a_maximised_window_has_no_edge_to_grab(dialog):
    dialog.setWindowState(Qt.WindowMaximized)
    assert glass._edges_at(dialog, QPoint(1, 1)) == Qt.Edge(0)


def test_a_point_outside_the_window_is_not_an_edge(dialog):
    assert glass._edges_at(dialog, QPoint(-5, 120)) == Qt.Edge(0)


# ---------------------------------------------------------------------------
# Who keeps a press
# ---------------------------------------------------------------------------

def test_selectable_text_keeps_the_press(dialog):
    label = QLabel("copy me", dialog)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    assert glass._owns_mouse_gesture(label, dialog) is True


def test_a_widget_holding_a_selection_keeps_the_press(dialog):
    class Selected(QWidget):
        def hasSelectedText(self):  # noqa: N802 - Qt naming
            return True

    assert glass._owns_mouse_gesture(Selected(dialog), dialog) is True


def test_a_widget_that_claims_its_gestures_keeps_the_press(dialog):
    surface = QWidget(dialog)
    surface.setProperty("spacrOwnsMouseGesture", True)
    assert glass._owns_mouse_gesture(surface, dialog) is True


def test_a_plain_label_inside_a_plain_panel_does_not(dialog):
    panel = QWidget(dialog)
    label = QLabel("just words", panel)
    assert glass._owns_mouse_gesture(label, dialog) is False


# ---------------------------------------------------------------------------
# Resizing from each edge
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("press,drag,expected", [
    ((1, 120), (-40, 0), QRect(60, 100, 360, 240)),
    ((318, 120), (40, 0), QRect(100, 100, 360, 240)),
    ((160, 1), (0, -30), QRect(100, 70, 320, 270)),
    ((160, 238), (0, 30), QRect(100, 100, 320, 270)),
])
def test_dragging_an_edge_moves_that_edge_only(dialog, monkeypatch, press,
                                               drag, expected):
    _off_wayland(monkeypatch)
    watcher = glass._ResizeByEdge(dialog)
    start = (500, 500)
    assert watcher.eventFilter(dialog, _press(press, glob=start)) is True
    end = (start[0] + drag[0], start[1] + drag[1])
    assert watcher.eventFilter(dialog, _move(press, glob=end)) is True
    assert dialog.geometry() == expected


def test_a_move_after_a_lost_release_ends_the_resize(dialog, monkeypatch):
    _off_wayland(monkeypatch)
    watcher = glass._ResizeByEdge(dialog)
    assert watcher.eventFilter(dialog, _press((318, 120))) is True
    before = dialog.geometry()

    assert watcher.eventFilter(
        dialog, _move((318, 120), glob=(600, 500), buttons=Qt.NoButton)) is False
    assert dialog.geometry() == before
    assert watcher.eventFilter(
        dialog, _move((318, 120), glob=(650, 500))) is False
    assert dialog.geometry() == before


def test_leaving_the_window_mid_resize_keeps_the_edge_lit(dialog, monkeypatch):
    _off_wayland(monkeypatch)
    watcher = glass._ResizeByEdge(dialog)
    watcher.eventFilter(dialog, _press((1, 120)))
    assert watcher._hint.edges == Qt.Edge.LeftEdge

    watcher.eventFilter(dialog, QEvent(QEvent.Type.Leave))
    assert watcher._hint.edges == Qt.Edge.LeftEdge


def test_on_wayland_the_compositor_resizes(dialog, monkeypatch):
    _on_wayland(monkeypatch)
    watcher = glass._ResizeByEdge(dialog)
    before = dialog.geometry()

    assert watcher.eventFilter(dialog, _press((318, 238))) is True
    assert dialog.asked_resize == [Qt.Edge.RightEdge | Qt.Edge.BottomEdge]
    watcher.eventFilter(dialog, _move((318, 238), glob=(600, 600)))
    assert dialog.geometry() == before


# ---------------------------------------------------------------------------
# Dragging the background
# ---------------------------------------------------------------------------

def test_a_drag_move_after_a_lost_release_does_not_move(dialog, monkeypatch):
    _off_wayland(monkeypatch)
    dragger = glass._DragByBackground(dialog)
    dragger.eventFilter(dialog, _press((160, 120), glob=(500, 500)))
    before = dialog.pos()

    assert dragger.eventFilter(
        dialog, _move((160, 120), glob=(560, 560), buttons=Qt.NoButton)) is False
    assert dragger.eventFilter(dialog, _move((160, 120), glob=(600, 600))) is False
    assert dialog.pos() == before


def test_a_jiggle_below_the_drag_distance_does_not_move(dialog, monkeypatch):
    _off_wayland(monkeypatch)
    dragger = glass._DragByBackground(dialog)
    dragger.eventFilter(dialog, _press((160, 120), glob=(500, 500)))
    before = dialog.pos()

    assert dragger.eventFilter(dialog, _move((160, 120), glob=(501, 500))) is False
    assert dialog.pos() == before


def test_on_wayland_the_compositor_moves_the_dialog(qtbot, monkeypatch):
    _on_wayland(monkeypatch)
    dialog = _HandledDialog(move_accepted=True)
    qtbot.addWidget(dialog)
    QVBoxLayout(dialog)
    dialog.setGeometry(100, 100, 320, 240)
    dragger = glass._DragByBackground(dialog)
    dragger.eventFilter(dialog, _press((160, 120), glob=(500, 500)))
    before = dialog.pos()

    assert dragger.eventFilter(dialog, _move((160, 120), glob=(560, 560))) is True
    assert dialog.asked_move == [True]
    assert dialog.pos() == before
    assert dragger.eventFilter(dialog, _move((160, 120), glob=(600, 600))) is False
    assert dialog.pos() == before


# ---------------------------------------------------------------------------
# A child on the edge
# ---------------------------------------------------------------------------

def test_a_press_on_a_child_at_the_edge_starts_the_resize(dialog, monkeypatch):
    _off_wayland(monkeypatch)
    dialog.layout().setContentsMargins(0, 0, 0, 0)
    panel = QWidget()
    dialog.layout().addWidget(panel)
    dialog.show()
    panel.setGeometry(0, 0, 320, 240)
    assert glass.let_the_user_resize(dialog) is True
    installer = glass._GlassInstaller()

    window_origin = dialog.mapToGlobal(QPoint(0, 0))
    glob = QPointF(window_origin.x() + 1, window_origin.y() + 120)
    press = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(1, 120), glob,
                        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    assert installer.eventFilter(panel, press) is True
    assert dialog._spacr_resizer._hint.edges == Qt.Edge.LeftEdge
