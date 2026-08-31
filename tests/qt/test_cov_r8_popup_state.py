"""`popup_state` — what the backdrop asks before it moves.

The flicker this exists to stop is a real one: a menu or a tooltip
composited over the native GL backdrop made the widgets around it
repaint, and the user saw the dock and the header flicker. Holding the
animation still while a popup is up takes a menu from 238 painted
widgets to 5 and a tooltip from 90 to 0.

So these tests are about the two things that can silently undo that: the
tooltip half being forgotten (a tooltip is NOT an activePopupWidget, and
watching only menus would have left half the bug in place), and the
helper being allowed to raise into an animation tick.
"""
from __future__ import annotations

import pytest

from spacr.qt.widgets import popup_state
from spacr.qt.widgets.popup_state import a_popup_is_on_screen


def test_nothing_is_showing_so_the_backdrop_may_move(qtbot):
    """The ordinary case: no popup, and the animation is not held."""
    from PySide6.QtWidgets import QWidget

    plain = QWidget()
    qtbot.addWidget(plain)
    plain.show()
    assert a_popup_is_on_screen() is False


def test_an_open_menu_holds_the_backdrop_still(qtbot):
    """A menu is an activePopupWidget, which is the half Qt makes easy."""
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QApplication, QMenu, QWidget

    host = QWidget()
    qtbot.addWidget(host)
    host.resize(200, 120)
    host.show()
    qtbot.waitExposed(host)

    menu = QMenu(host)
    menu.addAction("something")
    assert a_popup_is_on_screen() is False

    menu.popup(host.mapToGlobal(QPoint(10, 10)))
    try:
        qtbot.waitUntil(lambda: QApplication.activePopupWidget() is not None,
                        timeout=2000)
        assert a_popup_is_on_screen() is True
    finally:
        menu.close()
    qtbot.waitUntil(lambda: QApplication.activePopupWidget() is None,
                    timeout=2000)
    assert a_popup_is_on_screen() is False


def test_a_tooltip_counts_even_though_it_is_not_a_popup_widget(monkeypatch):
    """THE HALF THAT IS EASY TO MISS.

    A tooltip is its own `Qt::ToolTip` window and is NOT reported by
    `activePopupWidget`, so an implementation that asked only that
    question would answer "no popup" with a tooltip on screen -- and the
    tooltip flicker was reported alongside the menu one.
    """
    from PySide6.QtWidgets import QApplication, QToolTip

    monkeypatch.setattr(QApplication, "activePopupWidget",
                        staticmethod(lambda: None))
    monkeypatch.setattr(QToolTip, "isVisible", staticmethod(lambda: True))
    assert a_popup_is_on_screen() is True


def test_a_tooltip_that_has_gone_releases_the_backdrop(monkeypatch):
    """`isVisible` returning a false-y value must not read as showing."""
    from PySide6.QtWidgets import QApplication, QToolTip

    monkeypatch.setattr(QApplication, "activePopupWidget",
                        staticmethod(lambda: None))
    monkeypatch.setattr(QToolTip, "isVisible", staticmethod(lambda: False))
    assert a_popup_is_on_screen() is False


def test_a_question_that_cannot_be_answered_is_answered_no(monkeypatch):
    """Called from an animation tick, so it may never raise.

    Answering "no popup" is the behaviour the application had before this
    existed: the backdrop keeps animating. Raising would take the tick
    with it, and vispy retries a failed handler -- which is where the
    2,4,8...4096 repeat storm came from.
    """
    from PySide6.QtWidgets import QApplication

    def explode():
        raise RuntimeError("the C++ object is already gone")

    monkeypatch.setattr(QApplication, "activePopupWidget",
                        staticmethod(explode))
    assert a_popup_is_on_screen() is False


def test_the_tooltip_question_may_also_fail_without_taking_the_tick(
        monkeypatch):
    """The second call is inside the same guard as the first."""
    from PySide6.QtWidgets import QApplication, QToolTip

    def explode():
        raise RuntimeError("no tooltip machinery on this platform")

    monkeypatch.setattr(QApplication, "activePopupWidget",
                        staticmethod(lambda: None))
    monkeypatch.setattr(QToolTip, "isVisible", staticmethod(explode))
    assert a_popup_is_on_screen() is False


def test_it_is_asked_per_frame_and_not_per_event(monkeypatch):
    """The cost is the reason this is a poll rather than an event filter.

    The event-filter version ran Python for every event in the
    application -- 13,646 calls a second while a module opens -- and cost
    130 ms on the GUI-thread block that opening a module had just been
    fixed to shorten. This asks Qt two questions and returns.
    """
    calls = {"n": 0}
    from PySide6.QtWidgets import QApplication, QToolTip

    def counted():
        calls["n"] += 1
        return None

    monkeypatch.setattr(QApplication, "activePopupWidget",
                        staticmethod(counted))
    monkeypatch.setattr(QToolTip, "isVisible", staticmethod(lambda: False))
    for _ in range(10):
        a_popup_is_on_screen()
    assert calls["n"] == 10


class TestTheAnimationTicksActuallyAskIt:
    """The helper is only useful if the ticks consult it."""

    def test_the_gpu_fractal_tick_consults_it(self):
        import inspect

        from spacr.qt.widgets import fractal_travel

        src = inspect.getsource(fractal_travel)
        assert "a_popup_is_on_screen()" in src, (
            "the GPU backdrop stopped asking whether a popup is up")

    def test_the_ambient_engine_tick_consults_it(self):
        import inspect

        from spacr.qt.widgets import ambient

        src = inspect.getsource(ambient)
        assert "a_popup_is_on_screen()" in src, (
            "the ambient backdrop stopped asking whether a popup is up")
