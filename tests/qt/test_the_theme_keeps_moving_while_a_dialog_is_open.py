"""The backdrop keeps animating while Preferences or Help is up (385).

The report was "when the preferences menue or help menu is open the theme
in the background stops, the theme should continue." The cause was an
early return in ``AmbientWidget._on_tick`` that held the animation still
whenever Qt reported a popup on screen -- so an open menu froze the theme,
and so did any popup a dialog put up.

These tests are the number the instruction asked for rather than an
impression: they count frames over a fixed interval with the surface shut
and with it open, and assert the second interval is not zero.

THE ONE CASE THAT MUST STILL STOP is a pipeline run. That is a deliberate
pause through ``set_animating(False)``, and it is asserted here too, so a
future change that revives the popup hold by widening the pause cannot
pass by making the first test green.
"""

import time

import pytest
from PySide6.QtCore import QEventLoop, QPoint
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QMenu, QWidget

from spacr.qt.widgets.ambient import AmbientWidget


def _spin(seconds: float) -> None:
    """Run the event loop for a wall-clock interval, as the app would."""
    app = QApplication.instance()
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 5)


def _backdrop(qtbot):
    """A shown ambient backdrop with something drawn in front of it."""
    host = QWidget()
    QLabel("in front", parent=host)
    back = AmbientWidget(parent=host, theme="blobs")
    back.follow_parent()
    host.resize(320, 240)
    qtbot.addWidget(host)
    host.show()
    qtbot.waitExposed(host)
    back.show()
    _spin(0.3)
    return host, back


def _frames_over(back, seconds: float) -> int:
    """Frames the backdrop advanced in `seconds`."""
    before = back.frames_painted
    _spin(seconds)
    return back.frames_painted - before


class TestTheThemeKeepsMoving:

    def test_an_open_menu_does_not_stop_the_theme(self, qtbot):
        host, back = _backdrop(qtbot)
        shut = _frames_over(back, 0.5)
        assert shut > 0, "the backdrop was not running before the menu"

        menu = QMenu(host)
        menu.addAction("Keyboard shortcuts")
        menu.addAction("Set spaCR up again…")
        qtbot.addWidget(menu)
        menu.popup(host.mapToGlobal(QPoint(8, 8)))
        qtbot.waitUntil(lambda: menu.isVisible(), timeout=2000)

        assert _frames_over(back, 0.5) > 0, (
            "the theme froze while a menu was open, which is 385")
        menu.close()

    def test_an_open_dialog_does_not_stop_the_theme(self, qtbot):
        host, back = _backdrop(qtbot)
        assert _frames_over(back, 0.5) > 0

        dlg = QDialog(host)
        dlg.setModal(True)
        qtbot.addWidget(dlg)
        dlg.show()
        qtbot.waitExposed(dlg)

        assert _frames_over(back, 0.5) > 0, (
            "the theme froze behind a modal dialog -- Preferences is where "
            "the theme's own controls live, so this is the case that "
            "matters most")
        dlg.close()


class TestTheOneCaseThatStillStops:

    def test_a_run_still_pauses_the_backdrop(self, qtbot):
        """The pipeline pause is deliberate and must survive 385."""
        _host, back = _backdrop(qtbot)
        assert _frames_over(back, 0.4) > 0

        back.set_animating(False)
        assert _frames_over(back, 0.4) == 0, (
            "a pipeline run no longer pauses the backdrop")

        back.set_animating(True)
        assert _frames_over(back, 0.4) > 0, (
            "the backdrop did not come back after the run")
