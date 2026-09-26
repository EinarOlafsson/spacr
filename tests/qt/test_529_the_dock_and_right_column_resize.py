"""Item 529: the dock and the right-hand column resize; Ctrl + scroll sizes text.

The maintainer, 2026-09-25: "the user should be able to modify the width of
the dock when not hidden and the width of the pannels to the right (not
individually but one together.)for the pannels to the right holding ctrl and
scrolling should increase or decrease the font size."
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, QSettings, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent, QWheelEvent                 # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QWidget        # noqa: E402


def _pump(n: int = 8) -> None:
    for _ in range(n):
        QApplication.processEvents()


@pytest.fixture
def prefs_file(tmp_path, monkeypatch):
    """A throwaway preferences store, and the column filter back at 100 %."""
    from spacr.qt import preferences as prefs
    from spacr.qt.first_run import mark_tour_seen

    path = tmp_path / "spacr-529.ini"
    monkeypatch.setattr(prefs, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    mark_tour_seen()
    column = _column_filter()
    column.reset()
    yield prefs
    column.reset()


def _column_filter():
    from spacr.qt.live_zoom import install_column_text_scale

    return install_column_text_scale()


def _drag(widget, dx: int) -> None:
    """Press on ``widget``, move ``dx`` pixels sideways, release."""
    y = widget.height() / 2
    start = QPointF(2, y)
    end = QPointF(2 + dx, y)
    for kind, local, button, buttons in (
            (QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton),
            (QEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton),
            (QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton)):
        glob = QPointF(widget.mapToGlobal(local.toPoint()))
        QApplication.sendEvent(widget, QMouseEvent(
            kind, local, glob, button, buttons, Qt.NoModifier))
    _pump()


def _wheel(widget, notches: float = 1, modifiers=Qt.ControlModifier) -> bool:
    """Send a wheel event to ``widget``; returns whether it was accepted."""
    local = QPointF(10, 10)
    event = QWheelEvent(local, QPointF(widget.mapToGlobal(QPoint(10, 10))),
                        QPoint(0, 0), QPoint(0, int(120 * notches)),
                        Qt.NoButton, modifiers, Qt.NoScrollPhase, False)
    QApplication.sendEvent(widget, event)
    _pump()
    return event.isAccepted()


def _deepest(widget):
    """The innermost visible widget at ``widget``'s top-left corner."""
    child = widget.childAt(QPoint(12, 12))
    return child if child is not None else widget


class TestTheDockWidth:

    @pytest.fixture
    def window(self, qtbot, qt_theme_applied, prefs_file):
        from spacr.qt.app import MainWindow

        prefs_file.set_dock_mode("locked")
        prefs_file.set_dock_width(0)
        win = MainWindow()
        qtbot.addWidget(win)
        win.resize(1440, 900)
        win.show()
        qtbot.waitExposed(win)
        _pump()
        yield win

    def test_dragging_the_edge_widens_the_dock_and_is_remembered(
            self, qtbot, window, prefs_file):
        dock, edge = window._sidebar, window._dock_edge
        assert edge.isVisible()
        before = dock.width()
        assert before == dock.fitting_width()
        _drag(edge, 80)
        assert dock.width() == before + 80
        assert prefs_file.get_dock_width() == before + 80
        _drag(edge, -40)
        assert dock.width() == before + 40
        assert prefs_file.get_dock_width() == before + 40

        from spacr.qt.app import MainWindow
        again = MainWindow()
        qtbot.addWidget(again)
        again.resize(1440, 900)
        again.show()
        _pump()
        assert again._sidebar.width() == before + 40

    def test_hiding_and_showing_keeps_the_width(self, window):
        dock, edge = window._sidebar, window._dock_edge
        _drag(edge, 60)
        wide = dock.width()
        window.apply_dock_mode("hidden")
        _pump()
        assert not edge.isVisible()
        assert not window._dock_slot.isVisible()
        window.apply_dock_mode("locked")
        _pump()
        assert edge.isVisible()
        assert dock.width() == wide
        dock.refresh_visibility()
        assert dock.width() == wide

    def test_the_width_stays_between_its_bounds(self, window, prefs_file):
        from spacr.qt.preferences import scaled_px

        dock, edge = window._sidebar, window._dock_edge
        _drag(edge, 5000)
        assert dock.width() == scaled_px(dock.DRAG_MAX)
        _drag(edge, -5000)
        assert dock.width() == scaled_px(dock.DRAG_MIN)
        assert prefs_file.get_dock_width() == scaled_px(dock.DRAG_MIN)

    def test_a_double_click_fits_the_names_again(self, window, prefs_file):
        dock, edge = window._sidebar, window._dock_edge
        _drag(edge, 90)
        glob = QPointF(edge.mapToGlobal(QPoint(2, 10)))
        QApplication.sendEvent(edge, QMouseEvent(
            QEvent.MouseButtonDblClick, QPointF(2, 10), glob, Qt.LeftButton,
            Qt.LeftButton, Qt.NoModifier))
        _pump()
        assert dock.width() == dock.fitting_width()
        assert prefs_file.get_dock_width() == 0

    def test_ctrl_0_resets_the_column_text_before_it_goes_home(
            self, window, prefs_file):
        window.open_module("measure")
        _pump(20)
        screen = window._stack.currentWidget()
        column = _column_filter()
        column.set_scale(1.3)
        column.restyle_all()
        from PySide6.QtTest import QTest

        target = screen._console._input
        target.setFocus()
        QTest.keyClick(target, Qt.Key_0, Qt.ControlModifier)
        _pump()
        assert column.scale() == 1.0
        assert window._stack.currentWidget() is screen
        QTest.keyClick(target, Qt.Key_0, Qt.ControlModifier)
        _pump()
        assert window._stack.currentWidget() is not screen


@pytest.fixture
def measure(qtbot, qt_theme_applied, prefs_file):
    """Measure at a laptop's size, its body splitter never dragged before."""
    from spacr.qt.preferences import set_folded_panel
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets.collapsible_splitter import set_pane_extents

    set_pane_extents("measure::body", {})
    for name in ("Console", "System", "Actions", "Settings"):
        set_folded_panel(f"measure/{name}", False)

    def build():
        screen = AppScreen("measure")
        qtbot.addWidget(screen)
        screen.resize(1500, 860)
        screen.show()
        qtbot.waitExposed(screen)
        _pump(20)
        return screen

    yield build


class TestTheRightColumnResizesAsOne:

    def test_its_left_edge_moves_every_panel_together(self, measure):
        from spacr.qt.widgets.collapsible_splitter import get_pane_extents

        screen = measure()
        body = screen._body_splitter
        assert body.widget(1) is screen._runtime_wrap
        panels = (screen._runtime_wrap, screen._runtime_splitter,
                  screen._console_wrap, screen._usage_card,
                  screen._actions_section)
        before = [w.width() for w in panels]
        _drag(body.handle(1), 150)
        after = [w.width() for w in panels]
        shrink = before[0] - after[0]
        assert shrink >= 100
        assert all(b - a == shrink for b, a in zip(before, after))
        assert get_pane_extents("measure::body")["Runtime"] == after[0]

        again = measure()
        assert abs(again._runtime_wrap.width() - after[0]) <= 2
        assert abs(again._console_wrap.width() - after[2]) <= 2


class TestCtrlWheelSizesTheColumnText:

    @staticmethod
    def _px(widget) -> int:
        return widget.font().pixelSize()

    def test_ctrl_wheel_grows_the_column_text_and_nothing_else(
            self, measure, prefs_file, qtbot):
        screen = measure()
        column = _column_filter()
        title = screen._console_card.title_label
        system = screen._usage_card.title_label
        run = screen._btn_run
        outside = [w for w in screen._settings_panel.findChildren(QLabel)
                   if w.isVisible()][:20]
        outside.append(screen._header)
        before = (self._px(title), self._px(system), self._px(run),
                  screen._console._font_pt)
        before_outside = [w.font().pixelSize() for w in outside]

        viewport = _deepest(screen._console)
        assert column.root_of(viewport) is screen._runtime_wrap
        assert _wheel(viewport, 2)
        _pump()
        assert column.scale() == pytest.approx(1.2)
        assert self._px(title) > before[0]
        assert self._px(system) > before[1]
        assert self._px(run) > before[2]
        assert screen._console._font_pt > before[3]
        assert [w.font().pixelSize() for w in outside] == before_outside

        qtbot.wait(600)
        assert prefs_file.get_runtime_text_scale() == pytest.approx(1.2)

        from spacr.qt.live_zoom import ColumnTextScale
        assert ColumnTextScale().scale() == pytest.approx(1.2)

        from PySide6.QtTest import QTest
        QTest.keyClick(screen._console._input, Qt.Key_0, Qt.ControlModifier)
        _pump()
        assert column.scale() == 1.0
        assert prefs_file.get_runtime_text_scale() == 1.0
        assert (self._px(title), self._px(system), self._px(run),
                screen._console._font_pt) == before

    def test_the_size_stays_within_its_limits(self, measure, prefs_file):
        from spacr.qt.preferences import (RUNTIME_TEXT_SCALE_MAX,
                                          RUNTIME_TEXT_SCALE_MIN)

        screen = measure()
        column = _column_filter()
        target = _deepest(screen._usage_card)
        _wheel(target, 40)
        assert column.scale() == RUNTIME_TEXT_SCALE_MAX
        _wheel(target, -80)
        assert column.scale() == RUNTIME_TEXT_SCALE_MIN

    def test_the_wheel_is_left_alone_without_ctrl_or_outside_the_column(
            self, measure):
        screen = measure()
        column = _column_filter()
        inside = _deepest(screen._console)
        assert not column.eventFilter(inside, QWheelEvent(
            QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, 120),
            Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
        outside = _deepest(screen._settings_panel)
        assert column.root_of(outside) is None
        _wheel(outside, 3)
        assert column.scale() == 1.0

    def test_a_canvas_that_zooms_and_the_z_gesture_keep_their_wheel(
            self, measure):
        screen = measure()
        column = _column_filter()

        class Canvas(QWidget):
            def wheelEvent(self, event):  # noqa: N802
                event.accept()

        canvas = Canvas(screen._usage_card.body)
        canvas.resize(40, 40)
        canvas.show()
        _wheel(canvas, 2)
        assert column.scale() == 1.0

        from spacr.qt.live_zoom import install_live_zoom
        live = install_live_zoom()
        live._held = True
        try:
            _wheel(_deepest(screen._console), 2)
        finally:
            live._held = False
            live._baseline = []
        assert column.scale() == 1.0


def test_the_font_sheet_scales_sizes_and_nothing_else():
    from spacr.qt.live_zoom import font_size_rules, scaled_font_sheet

    sheet = ("/* a */ QLabel#CardTitle { font-size: 15px; font-weight: 600; }"
             "\nQPushButton { color: red; }\nQWidget { font-size: 10pt; }")
    assert font_size_rules(sheet) == [("QLabel#CardTitle", 15.0, "px"),
                                      ("QWidget", 10.0, "pt")]
    out = scaled_font_sheet(sheet, 1.2)
    assert "QLabel#CardTitle { font-size: 18px; }" in out
    assert "QWidget { font-size: 12.0pt; }" in out
    assert "color" not in out and "weight" not in out
