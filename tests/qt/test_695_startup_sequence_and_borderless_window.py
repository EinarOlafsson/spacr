"""Item 695: the startup logo on one borderless window, its 0.5 + 4 s timeline
and the Help replay.

spaCR opens as ONE window, borderless and maximised to its screen's work
area (not full-screen mode). The first thing painted in it is the black
container with the logo photomosaic at a quarter of the screen height; the logo
holds for 0.5 s (longer while loading), blows away from 0.5 to 4.5 s, and the
container fades from 1.5 to 3.5 s. Help > Show startup animation plays it again.
The borderless window carries its own minimise, maximise/restore and close
marks, and a remembered switch gives back a framed window.

Every timing here is driven by a fake clock, so no test waits in real time.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QAction, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMenu, QToolButton  # noqa: E402

from spacr.qt.widgets import loading_screen as ls  # noqa: E402

pytestmark = pytest.mark.qt


class _Clock:
    """A clock the test moves by hand."""

    def __init__(self, now: float = 500.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def _visible_windows():
    return {w for w in QApplication.topLevelWidgets() if w.isVisible()}


@pytest.fixture()
def on_a_display(monkeypatch):
    from PySide6.QtGui import QGuiApplication

    monkeypatch.delenv(ls.NO_SPLASH_ENV, raising=False)
    monkeypatch.setattr(QGuiApplication, "platformName",
                        staticmethod(lambda: "xcb"))


@pytest.fixture()
def window(qtbot, qt_theme_applied):
    from spacr.qt.app import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()



@pytest.fixture()
def startup_finished(window, qtbot):
    """Let the real initial cover retire before asking the window to replay it."""
    from spacr.qt import app as app_mod

    app_mod._show_at_launch(window)
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: not window.findChildren(ls.LoadingScreen),
                    timeout=10000)



@pytest.fixture()
def paused_ambient(startup_finished, window):
    """Pause unrelated real backdrops before simulating an unreadable preference."""
    from spacr.qt import preferences
    from spacr.qt.widgets.ambient import AmbientWidget

    preferences.set_ambient_animation("none")
    for backdrop in window.findChildren(AmbientWidget):
        backdrop.set_animating(False)
        assert not backdrop.is_running()


def _menu(window, title):
    for menu in window.menuBar().findChildren(QMenu):
        if menu.title().replace("&", "") == title:
            return menu
    raise AssertionError(f"no {title} menu")


# --- the timeline ----------------------------------------------------------

def test_the_startup_timeline_on_one_borderless_window(
        qtbot, qt_theme_applied, monkeypatch, on_a_display):
    """The 695 acceptance test: one window, the logo, 0-0.5-1.5-3.5-4.5 s."""
    from spacr.qt import app as app_mod
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "get_preload_policy", lambda: "lazy")
    clock = _Clock()
    made = []

    def install(self):
        cover = ls.LoadingScreen(parent=self)
        cover._clock = clock
        cover.setGeometry(self.rect())
        cover.show_mosaic()
        cover.raise_()
        cover.show()
        made.append(cover)
        return cover

    monkeypatch.setattr(app_mod.MainWindow, "_install_loading_screen",
                        install)
    before = _visible_windows()
    window = app_mod.MainWindow()
    qtbot.addWidget(window)
    app_mod._open_at_the_measured_width(window)
    app_mod._show_at_launch(window)
    qtbot.waitExposed(window)
    cover = made[0]

    assert _visible_windows() - before == {window}, "one top-level window"
    assert not cover.isWindow() and cover.window() is window
    assert window.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert not window.windowState() & Qt.WindowState.WindowFullScreen
    assert window.isMaximized()
    screen = window.screen()
    qtbot.waitUntil(lambda: window.geometry() == screen.availableGeometry(),
                    timeout=3000)
    assert cover.geometry() == window.rect()

    first = window.grab().toImage()
    start = cover._first_paint_at
    assert start == clock.now, "the logo is the first thing painted"

    x, y, side, tile, ratio = cover._mosaic_layout()
    assert side == pytest.approx(screen.geometry().height() / 4.0)
    assert x + side / 2.0 == pytest.approx(window.width() / 2.0)
    assert y + side / 2.0 == pytest.approx(window.height() / 2.0)
    mosaic = cover._mosaic
    assert side * ratio <= tile * mosaic.grid, "decoded at or above its size"
    assert tile <= mosaic.tile_px, "the resource is sharp at this size"
    assert mosaic.tile_px * mosaic.grid >= 2160, "sharp at half of 4K at 2x"

    qtbot.waitUntil(lambda: window._loading_screen is None, timeout=3000)
    assert cover._state == "holding"
    assert not ls.blow_away_running(), "the window loads under the hold"
    for held in (0.1, 0.25, 0.49):
        clock.now = start + held
        cover._begin()
        assert cover._state == "holding", f"no tile moves at {held} s"
        assert window.grab().toImage() == first

    clock.now = start + 0.5
    cover._begin()
    assert cover._state == "playing" and cover.elapsed() == 0.0
    model = cover._model
    assert model.frame(0.0)[0].size == 0, "nothing moves at 0.5 s"
    assert ls.blow_away_running()

    opacity = []
    for seq in np.arange(0.5, 4.5001, 0.05):
        clock.now = start + float(seq)
        opacity.append((float(seq), cover.container_opacity()))
    assert all(o == 1.0 for s, o in opacity if s <= 1.5 + 1e-9)
    fading = [o for s, o in opacity if 1.5 + 1e-9 < s < 3.5 - 1e-9]
    assert fading and all(0.0 < o < 1.0 for o in fading)
    assert all(b < a for a, b in zip(fading, fading[1:])), "monotonic"
    assert all(o == 0.0 for s, o in opacity if s >= 3.5 - 1e-9)

    corner = (4, window.height() // 2)
    clock.now = start + 1.5
    cover._tick()
    assert window.grab().toImage().pixelColor(*corner).name() == "#000000"
    clock.now = start + 3.5
    cover._tick()
    shown = window.grab().toImage().pixelColor(*corner)
    cover.hide()
    bare = window.grab().toImage().pixelColor(*corner)
    cover.show()
    assert shown.name() == bare.name(), "fully visible at 3.5 s"

    assert float(np.max(model.fly_launch + model.life)) <= 4.0
    assert model.frame(4.0)[0].size == 0
    assert model.launched(4.0).size == model.grid ** 2
    clock.now = start + 4.49
    cover._tick()
    assert cover._state == "playing"
    clock.now = start + 4.5
    cover._tick()
    assert cover._state == "done" and cover.isHidden(), "no tile after 4.5 s"
    assert not ls.blow_away_running()


def test_the_logo_holds_longer_while_loading(host_cover):
    cover, clock = host_cover
    cover.grab()
    start = clock.now
    clock.now = start + 3.5
    assert cover.blow_away(force=True)
    cover._begin()
    assert cover._state == "playing"
    clock.now = start + 3.5 + 1.0
    assert cover.container_opacity() == 1.0
    clock.now = start + 3.5 + 3.0
    assert cover.container_opacity() == 0.0
    cover._finish()


def test_an_early_timer_waits_out_the_hold(host_cover, qtbot):
    cover, clock = host_cover
    cover.grab()
    assert cover.blow_away(force=True)
    clock.now += 0.4995
    cover._begin()
    assert cover._state == "holding"
    clock.now += 0.01
    qtbot.waitUntil(lambda: cover._state == "playing", timeout=2000)
    cover._finish()


@pytest.fixture()
def host_cover(qtbot):
    from PySide6.QtWidgets import QWidget

    host = QWidget()
    host.resize(1000, 700)
    qtbot.addWidget(host)
    cover = ls.LoadingScreen(total=2, parent=host)
    clock = _Clock()
    cover._clock = clock
    assert cover.show_mosaic()
    yield cover, clock
    host.close()


def test_the_strap_line_fades_out_early_in_the_blow_away(host_cover):
    cover, clock = host_cover
    cover.grab()
    assert cover.blow_away(force=True)
    clock.now += ls._MOSAIC_MIN_HOLD_S
    cover._begin()
    for t in (0.1, 0.5, 1.5, 3.9):
        clock.now = cover._began_at + t
        cover.grab()
    stats = cover.frame_stats()
    assert stats["frames"] == 4 and stats["most_tiles"] > 0
    cover._finish()
    assert cover.container_opacity() == 0.0


def test_the_container_fades_as_the_item_says():
    assert ls._container_opacity(0.0) == 1.0
    assert ls._container_opacity(1.0) == 1.0
    assert 0.0 < ls._container_opacity(2.0) < 1.0
    assert ls._container_opacity(3.0) == 0.0
    assert ls._container_opacity(0.0, plain=True) == 1.0
    assert ls._container_opacity(0.5, plain=True) == pytest.approx(0.5)
    assert ls._container_opacity(1.0, plain=True) == 0.0


def test_the_stretched_flight_ends_on_time():
    occupied = np.ones((80, 80), bool)
    model = ls.BlowAway(occupied, (0.0, 0.0), 500.0, sweep=ls._SWEEP_S,
                        pace=ls._PACE, end=ls._BLOW_AWAY_S)
    assert model.duration == ls._BLOW_AWAY_S
    assert float(np.max(model.fly_launch + model.life)) <= 4.0 - 0.02 + 1e-9
    assert model.launch.max() <= model.start + model.sweep
    quick = ls.BlowAway(occupied, (0.0, 0.0), 500.0)
    np.testing.assert_allclose(model.life, np.minimum(
        quick.life * ls._PACE, 4.0 - 0.02 - model.fly_launch))
    first = int(np.argmin(model.fly_launch))
    t = float(model.fly_launch[first]) + 0.2
    slow = model.frame(t, floor=0.0)
    at = int(np.flatnonzero(slow[0] == first)[0])
    fast = quick.frame(float(quick.fly_launch[first]) + 0.2 / ls._PACE,
                       floor=0.0)
    at_fast = int(np.flatnonzero(fast[0] == first)[0])
    assert slow[1][at] == pytest.approx(fast[1][at_fast])


def test_a_cover_without_a_screen_sizes_the_logo_by_itself(host_cover):
    cover, _clock = host_cover
    cover.screen = lambda: None
    x, y, side, tile, ratio = cover._mosaic_layout()
    assert side == pytest.approx(cover.height() / 4.0)


# --- Help > Show startup animation ----------------------------------------

@pytest.mark.usefixtures("startup_finished")
def test_help_replays_the_sequence_and_leaves_the_window_as_it_was(
        window, qtbot):
    from spacr.qt import app as app_mod

    app_mod._show_at_launch(window)
    qtbot.waitExposed(window)
    QApplication.processEvents()
    action = window.findChild(QAction, "ShowStartupAnimationAction")
    assert action is not None
    assert action in _menu(window, "Help").actions()
    assert action.text() == "Show startup animation"
    geometry, state = window.geometry(), window.windowState()
    flags, focus = window.windowFlags(), QApplication.focusWidget()

    action.trigger()
    covers = [c for c in window.findChildren(ls.LoadingScreen)
              if not c.isHidden()]
    assert len(covers) == 1
    cover = covers[0]
    clock = _Clock()
    cover._clock = clock
    assert cover.geometry() == window.rect()
    assert cover.mosaic_shown() and cover.lit_phases() == 3
    assert ls.replay_startup_animation(window) is None, "one at a time"
    cover.grab()
    assert cover._state == "holding"
    clock.now += 0.5
    cover._begin()
    assert cover._state == "playing" and cover._model is not None
    clock.now += 4.0
    cover._tick()
    assert cover._state == "done"
    qtbot.waitUntil(lambda: not window.findChildren(ls.LoadingScreen),
                    timeout=2000)
    assert window.geometry() == geometry
    assert window.windowState() == state
    assert window.windowFlags() == flags
    assert QApplication.focusWidget() is focus


@pytest.mark.usefixtures("startup_finished")
def test_the_replay_plays_with_the_startup_switch_off(window):
    from spacr.qt import preferences

    preferences.set_startup_animation(False)
    cover = ls.replay_startup_animation(window)
    assert cover is not None and cover._state in ("waiting", "holding")
    cover._finish()


@pytest.mark.usefixtures("startup_finished")
def test_reduce_motion_holds_then_cross_fades(window, qtbot):
    from spacr.qt import preferences

    preferences.set_reduce_motion(True)
    cover = ls.replay_startup_animation(window)
    clock = _Clock()
    cover._clock = clock
    cover.grab()
    clock.now += 0.5
    cover._begin()
    assert cover._state == "playing" and cover._model is None
    assert cover._tiles == []
    for t, expected in ((0.0, 1.0), (0.5, 0.5), (0.99, None)):
        clock.now = cover._began_at + t
        cover.grab()
        if expected is not None:
            assert cover.container_opacity() == pytest.approx(expected)
    assert cover.frame_stats()["most_tiles"] == 0
    clock.now = cover._began_at + 1.0
    cover._tick()
    assert cover._state == "done"


@pytest.mark.usefixtures("paused_ambient")
def test_an_unreadable_reduce_motion_counts_as_on(window, monkeypatch):
    from spacr.qt import preferences

    def broken():
        raise RuntimeError("store gone")

    monkeypatch.setattr(preferences, "get_reduce_motion", broken)
    cover = ls.replay_startup_animation(window)
    assert cover._plain is True
    cover._finish()


def test_no_replay_without_the_logo(window, monkeypatch):
    monkeypatch.setattr(ls.MosaicMap, "load", classmethod(lambda cls: None))
    assert ls.replay_startup_animation(window) is None


@pytest.mark.usefixtures("startup_finished")
def test_a_hidden_plain_cover_does_not_block_the_replay(window):
    plain = ls.LoadingScreen(parent=window)
    plain.hide()
    cover = ls.replay_startup_animation(window)
    assert cover is not None
    cover._finish()


# --- the window controls ---------------------------------------------------

def _marks(window):
    corner = window.menuBar().cornerWidget(Qt.Corner.TopRightCorner)
    return {b.objectName(): b for b in corner.findChildren(QToolButton)}


def test_the_top_bar_minimises_maximises_restores_and_closes(window, qtbot):
    from spacr.qt import app as app_mod

    corner = window.menuBar().cornerWidget(Qt.Corner.TopRightCorner)
    names = [b.objectName() for b in corner.findChildren(QToolButton)]
    assert names == ["MinimiseWindow", "MaximiseWindow", "FullScreenToggle",
                     "CloseWindow"]
    window.resize(1300, 800)
    window.show()
    qtbot.waitExposed(window)
    mark = _marks(window)["MaximiseWindow"]
    assert mark.toolTip() == "Maximise, or restore the size before"
    resting = mark.icon().pixmap(18, 18).toImage()
    mark.click()
    qtbot.waitUntil(window.isMaximized, timeout=2000)
    assert mark.icon().pixmap(18, 18).toImage() != resting, "restore mark"
    mark.click()
    qtbot.waitUntil(lambda: not window.isMaximized(), timeout=2000)
    assert app_mod.CHROME_HOVER["MaximiseWindow"] == "#3C82DC"


def test_the_two_maximise_marks_differ():
    from spacr.qt.app import MainWindow

    square = MainWindow._maximise_icon(18).pixmap(18, 18).toImage()
    pair = MainWindow._maximise_icon(18, "#3C82DC",
                                     restore=True).pixmap(18, 18).toImage()
    assert square != pair


def _press(x=600.0, y=4.0):
    return QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(x, y),
                       QPointF(x, y), Qt.MouseButton.LeftButton,
                       Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier)


def test_a_framed_window_leaves_dragging_and_resizing_to_its_frame(
        window, qtbot):
    window.resize(1300, 800)
    window.show()
    qtbot.waitExposed(window)
    bar = window.menuBar()
    calls = []
    window.windowHandle().startSystemMove = lambda: calls.append(1) or True
    window._menu_drag.eventFilter(bar, _press())
    window._set_borderless(False)
    assert window._menu_drag.eventFilter(bar, _press()) is False
    assert window.eventFilter(bar, _press()) is False
    resizer = window._spacr_resizer
    edge = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(1, 400),
                       QPointF(1, 400), Qt.MouseButton.LeftButton,
                       Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier)
    assert resizer.eventFilter(window, edge) is False
    window._set_borderless(True)


# --- the framed / borderless switch ----------------------------------------

def test_the_menu_switch_keeps_a_normal_window_visible_and_placed(window, qtbot):
    from PySide6.QtCore import QPoint

    action = window.findChild(QAction, "BorderlessWindowAction")
    action.setChecked(True)
    window.resize(1300, 800)
    window.move(70, 90)
    window.showNormal()
    qtbot.waitExposed(window)
    geometry = window.geometry()
    for borderless in (False, True):
        menu = _menu(window, "Window")
        menu.popup(window.mapToGlobal(QPoint(200, 70)))
        qtbot.waitUntil(menu.isVisible)
        qtbot.mouseClick(menu, Qt.MouseButton.LeftButton,
                         pos=menu.actionGeometry(action).center())
        qtbot.waitUntil(lambda: window.isVisible()
                       and QApplication.activeWindow() is window)
        assert not menu.isVisible()
        assert bool(window.windowFlags() & Qt.FramelessWindowHint) is borderless
        assert window.geometry() == geometry
        assert window.normalGeometry() == geometry
        assert not window.isMaximized() and not window.isFullScreen()


def test_the_switch_gives_a_framed_window_and_is_remembered(window, qtbot):
    from spacr.qt import preferences

    window.resize(1300, 800)
    window.show()
    qtbot.waitExposed(window)
    window.showMaximized()
    qtbot.waitUntil(window.isMaximized, timeout=2000)
    action = window.findChild(QAction, "BorderlessWindowAction")
    assert action.isCheckable() and action.isChecked()
    assert action in _menu(window, "Window").actions()
    assert action in _menu(window, "spaCR").actions()

    action.setChecked(False)
    assert preferences.get_borderless_window() is False
    assert not window.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert window.isVisible() and window.isMaximized()
    marks = _marks(window)
    assert all(marks[n].isHidden() for n in
               ("MinimiseWindow", "MaximiseWindow", "CloseWindow"))
    assert not marks["FullScreenToggle"].isHidden()

    action.setChecked(True)
    assert preferences.get_borderless_window() is True
    assert window.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert not marks["CloseWindow"].isHidden()
    window._set_borderless(True)
    assert window._borderless is True


def test_a_framed_choice_opens_framed(qtbot, qt_theme_applied):
    from spacr.qt import app as app_mod
    from spacr.qt import preferences

    preferences.set_borderless_window(False)
    window = app_mod.MainWindow()
    qtbot.addWidget(window)
    assert not window.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert window.findChild(QAction, "BorderlessWindowAction").isChecked() \
        is False
    assert all(b.isHidden() for b in window._frame_marks())
    app_mod._show_at_launch(window)
    assert window.isVisible() and not window.isMaximized()
    window._set_borderless(True)
    window.hide()
    window._set_borderless(False)
    assert not window.isVisible(), "a hidden window stays hidden"
    window.close()


def test_a_minimum_larger_than_the_screen_is_lowered_to_fit(window):
    from PySide6.QtCore import QSize

    from spacr.qt import app as app_mod

    work = window.screen().availableGeometry().size()
    window.setMinimumSize(work + QSize(400, 300))
    app_mod._show_at_launch(window)
    from spacr.qt.gui_scale import _actual_minimum_size

    assert _actual_minimum_size(window) == work
    assert window.isMaximized()


def test_the_launch_survives_a_window_without_a_screen(window):
    from spacr.qt import app as app_mod

    window.screen = lambda: None
    app_mod._show_at_launch(window)
    assert window.isMaximized()


def test_the_borderless_preference_defaults_on():
    from spacr.qt import preferences

    assert preferences.get_borderless_window() is True
    preferences.set_borderless_window(False)
    assert preferences.get_borderless_window() is False


@pytest.mark.parametrize('plain', [False, True])
def test_mosaic_hold_and_initial_departure_have_no_text_or_rules(host_cover, monkeypatch, plain):
    cover, clock = host_cover
    from PySide6.QtGui import QImage

    def assert_clear_below_logo(image):
        _x, y, side, _tile, _ratio = cover._mosaic_layout()
        start = int(y + side + 10)
        image = image.toImage().convertToFormat(QImage.Format_RGBA8888)
        values = np.frombuffer(image.bits(), np.uint8).reshape(image.height(), image.width(), 4)
        band = values[start:, :, :3]
        assert band.size
        assert np.all(band == band[-1, -1])

    def unexpected_strap(*args):
        raise AssertionError('Mosaic rendering must not draw the plain-fallback strap.')

    monkeypatch.setattr(cover, '_paint_strap', unexpected_strap)
    assert_clear_below_logo(cover.grab())
    assert cover.blow_away(force=True, plain=plain)
    clock.now += ls._MOSAIC_MIN_HOLD_S
    cover._begin()
    clock.now = cover._began_at
    assert_clear_below_logo(cover.grab())
    cover._finish()
