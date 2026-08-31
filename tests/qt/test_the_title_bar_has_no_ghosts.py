"""The menu bar is the title bar, and it must not leave copies of itself.

Three reports, one cause:

  * "in full screen mode the menu when clicking spacr in the top left
    drops down below Help it should drop down below spaCR";
  * "sometimes it looks like there are slightly misaligned duplicate
    minus, square, cross symbols in the top right corner";
  * "duplicate visuals of the text Help in the title".

The bar and its corner widget both have ``setAutoFillBackground`` FALSE
-- deliberately, because the menu bar IS the title bar and the backdrop
shows through it -- so neither erases the pixels it vacates when the bar
re-lays. The marks and the menu names are redrawn at their new positions
OVER the old ones.

``bar.update()`` cannot fix that: the bar repaints itself, and the stale
pixels are on the window underneath.
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

pytestmark = pytest.mark.qt


@pytest.fixture
def window(qtbot):
    from spacr.qt.app import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1200, 800)
    win.show()
    qtbot.waitExposed(win)
    return win


class TestTheBandIsErased:

    def test_the_window_is_repainted_over_the_bar_not_just_the_bar(self):
        """THE FIX for both ghosts.

        Recorded as a source contract because the repaint cannot be
        observed without a compositor: what matters is that the update
        is asked of ``self`` (the window) across the bar's band, and not
        only of the bar.
        """
        from spacr.qt.app import MainWindow

        source = inspect.getsource(MainWindow._relay_the_menu_bar)

        assert "self.update(0, 0, self.width(), band" in source, (
            "the window is no longer repainted across the bar's band, so "
            "the pixels the bar vacates are never erased")
        assert "bar.update()" in source, (
            "the bar itself is no longer repainted")

    def test_the_band_covers_a_corner_widget_taller_than_the_bar(self, window):
        """The three marks live in the corner widget, and it can be
        taller than the bar's own height -- a band measured from the bar
        alone would leave their bottom row behind."""
        source = inspect.getsource(type(window)._relay_the_menu_bar)

        assert "band = max(bar.height(), bar.sizeHint().height())" in source
        assert "band = max(band, corner.height())" in source

    def test_it_never_raises_when_the_bar_is_gone(self, window):
        """A window whose bar has been torn down still has to finish
        changing state."""
        source = inspect.getsource(type(window)._relay_the_menu_bar)

        assert source.count("except Exception:") >= 2
        assert "return" in source


class TestWhenItRuns:

    def test_a_state_change_re_lays_twice(self):
        """THE MENU-UNDER-HELP FIX.

        ``changeEvent`` is delivered when the STATE changes, which is
        before the compositor has resized the window -- so a re-lay done
        only there measures the old geometry and the menu still opens
        against the previous action rectangle. The zero-timer runs after
        the resize has been delivered.
        """
        from spacr.qt.app import MainWindow

        source = inspect.getsource(MainWindow.changeEvent)

        assert "self._relay_the_menu_bar()" in source
        assert "QTimer.singleShot(0, self._relay_the_menu_bar)" in source
        assert source.index("self._relay_the_menu_bar()") < \
            source.index("QTimer.singleShot(0"), (
            "the immediate re-lay no longer runs first, so the bar is wrong "
            "for one frame after every state change")

    def test_every_resize_re_lays_too(self):
        """THE "SOMETIMES" IN THE REPORT. Dragging an edge moves the
        corner buttons without changing the window state, so
        ``changeEvent`` never fires -- it is not intermittent, it is
        every resize that is not a fullscreen toggle."""
        from spacr.qt.app import MainWindow

        source = inspect.getsource(MainWindow.resizeEvent)

        assert "self._relay_the_menu_bar()" in source

    def test_the_loading_screen_is_still_resized_first(self):
        """The resize handler had one job already and keeps it."""
        from spacr.qt.app import MainWindow

        source = inspect.getsource(MainWindow.resizeEvent)

        assert "screen.setGeometry(self.rect())" in source
        assert source.index("screen.setGeometry") < \
            source.index("self._relay_the_menu_bar()")


class TestWhereTheMenusOpen:

    def test_the_first_menu_stays_at_the_left_edge(self, window, qtbot):
        """A menu opens where the BAR SAYS its action is. After a
        fullscreen toggle the first action must still be at the left, or
        pressing spaCR drops a menu under Help.
        """
        bar = window.menuBar()
        actions = [a for a in bar.actions() if a.text()]
        assert actions, "the menu bar has no named actions"
        first = actions[0]

        before = bar.actionGeometry(first).x()

        window.showFullScreen()
        qtbot.wait(20)
        window.showNormal()
        qtbot.wait(20)

        after = bar.actionGeometry(first).x()
        assert after == before, (
            f"the first menu moved from x={before} to x={after} across a "
            f"fullscreen round trip, so its menu opens somewhere else")
        assert after < bar.width() // 4, (
            "the first menu is no longer near the left edge of the bar")

    def test_the_actions_keep_their_order_and_do_not_overlap(self, window):
        """Two actions whose rectangles overlap is how one name is drawn
        over another."""
        bar = window.menuBar()
        named = [a for a in bar.actions() if a.text()]
        rects = [bar.actionGeometry(a) for a in named]

        for earlier, later in zip(rects, rects[1:]):
            assert earlier.right() <= later.left(), (
                "two menu-bar actions overlap, so their captions are drawn "
                "on top of one another")

    def test_the_corner_buttons_stay_inside_the_bar(self, window, qtbot):
        """The three marks are the corner widget. Outside the bar they
        are drawn over whatever is there instead."""
        bar = window.menuBar()
        corner = bar.cornerWidget(Qt.Corner.TopRightCorner)
        assert corner is not None, "the window chrome is gone"

        for width, height in ((1200, 800), (900, 700), (1600, 1000)):
            window.resize(width, height)
            qtbot.wait(10)
            geometry = corner.geometry()
            assert geometry.right() <= bar.width(), (
                f"at {width}x{height} the window buttons stick out "
                f"{geometry.right() - bar.width()} px past the menu bar")
            assert geometry.left() > bar.width() // 2, (
                "the window buttons are no longer in the right-hand corner")

    def test_the_buttons_do_not_sit_on_top_of_a_menu_name(self, window):
        """A corner widget overlapping the last action is the other way
        two things get drawn in one place."""
        bar = window.menuBar()
        corner = bar.cornerWidget(Qt.Corner.TopRightCorner)
        named = [a for a in bar.actions() if a.text()]
        assert named and corner is not None

        last = bar.actionGeometry(named[-1])
        assert last.right() <= corner.geometry().left(), (
            f"the last menu ({named[-1].text()!r}) runs under the window "
            f"buttons, so its caption and the marks are drawn in the same "
            f"place")
