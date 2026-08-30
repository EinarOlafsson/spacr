"""The window's failure ledges: what `spacr.qt.app` does when a part is missing.

The module is written so that any one optional piece -- the spaceout fractal,
the loading screen, the resize grips, the menu-bar corner widget, a plugin's
screen factory, an app whose module will not import -- can fail or simply not
be there without taking the window down with it. Those branches are the ones a
user only meets on the machine where something is already wrong, which is when
a crash is least affordable.

Every test drives the real :class:`MainWindow`, the real ``Sidebar`` and the
real registry offscreen; only the thing that is supposed to be broken is
faked. Each test also drives the working side of the same branch, so "nothing
happened" is asserted against a case where something did.
"""
from __future__ import annotations

import logging

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, QSize, Qt
from PySide6.QtGui import (QCloseEvent, QKeyEvent, QMouseEvent, QResizeEvent,
                           QWheelEvent, QWindowStateChangeEvent)
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QWidget

from spacr.qt import app as app_mod
from spacr.qt import i18n as i18n_mod
from spacr.qt import iconset, preferences, theme
from spacr.qt.app import MainWindow, Sidebar
from spacr.qt.widgets import fractal_travel


@pytest.fixture
def win(qtbot, qt_theme_applied):
    """A live MainWindow, cleaned up by pytest-qt."""
    window = MainWindow()
    qtbot.addWidget(window)
    return window


class _Recorder:
    """A stand-in that answers anything and remembers what it was asked.

    ``_Recorder(close=RuntimeError(...))`` raises that from ``close()`` while
    still recording the call, which is how a collaborator that is present but
    broken is told apart from one that was never reached.
    """

    def __init__(self, **behaviour):
        self.calls = []
        self._behaviour = behaviour

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def _call(*args, **kwargs):
            self.calls.append((name, args))
            outcome = self._behaviour.get(name)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return _call

    def names(self):
        return [name for name, _ in self.calls]


# --- the spaceout backdrop -------------------------------------------------

def test_a_backdrop_that_cannot_be_resized_does_not_swallow_the_resize(qtbot):
    """A dead fractal widget must not break the screen it sits behind.

    An exception out of an event filter propagates into Qt's dispatch, so a
    backdrop Qt has already destroyed would stop the screen resizing at all.
    """
    screen = QWidget()
    qtbot.addWidget(screen)
    screen.resize(320, 240)
    backdrop = QWidget(screen)
    event = QResizeEvent(QSize(320, 240), QSize(10, 10))
    ok = app_mod._FractalFollowsItsScreen(backdrop, screen)
    assert ok.eventFilter(screen, event) is False
    assert backdrop.geometry() == screen.rect()

    gone = _Recorder(setGeometry=RuntimeError("C++ object already deleted"))
    brittle = app_mod._FractalFollowsItsScreen(gone, screen)
    assert brittle.eventFilter(screen, event) is False
    assert gone.names() == ["setGeometry"], "it tried before giving up"


def test_a_fractal_that_cannot_be_placed_is_shut_down_not_left_running(
        qtbot, monkeypatch, caplog):
    """A half-installed fractal keeps a render thread with nothing on screen.

    The installer reports failure so the caller falls back to the ordinary
    ambient backdrop, and shuts the widget down first because its thread is
    already running. A shutdown that fails too must not crash the launch.
    """
    caplog.set_level(logging.DEBUG, logger="spacr.qt.app")
    screen = QWidget()
    qtbot.addWidget(screen)
    built = []
    monkeypatch.setattr(theme, "spaceout_enabled", lambda: False)
    monkeypatch.setattr(fractal_travel, "create_fractal_widget",
                        lambda *a, **k: built.append("built"))
    assert app_mod.install_the_spaceout_fractal(screen) is False
    assert built == [], "an ordinary launch must not build a fractal at all"

    monkeypatch.setattr(theme, "spaceout_enabled", lambda: True)
    tidy = _Recorder(setParent=RuntimeError("no such parent"))
    monkeypatch.setattr(fractal_travel, "create_fractal_widget",
                        lambda *a, **k: tidy)
    assert app_mod.install_the_spaceout_fractal(screen) is False
    assert tidy.names() == ["setParent", "shutdown"]

    stuck = _Recorder(setParent=RuntimeError("no such parent"),
                      shutdown=RuntimeError("the render thread is wedged"))
    monkeypatch.setattr(fractal_travel, "create_fractal_widget",
                        lambda *a, **k: stuck)
    assert app_mod.install_the_spaceout_fractal(screen) is False
    assert stuck.names() == ["setParent", "shutdown"]
    assert "Could not place the spaceout fractal" in caplog.text


# --- the registry ----------------------------------------------------------

def test_an_app_whose_screen_module_will_not_import_keeps_the_window_up(
        monkeypatch, caplog):
    """One broken screen costs that screen, not the whole application.

    Screen factories are import stand-ins. A module that cannot be imported
    here -- a missing optional dependency on this machine -- must hand back
    ``None`` so the app falls back to the generic settings screen, and must
    leave the stand-in in place so the next attempt fails the same way.
    """
    caplog.set_level(logging.ERROR, logger="spacr.qt.app")
    good, bad = "__cov_wf_good__", "__cov_wf_bad__"
    monkeypatch.setitem(app_mod.APP_FACTORIES, good,
                        app_mod.LazyScreenFactory("spacr.qt.app", "Sidebar"))
    monkeypatch.setitem(
        app_mod.APP_FACTORIES, bad,
        app_mod.LazyScreenFactory("spacr.qt._no_such_screen_module", "make"))
    assert app_mod.registered_factory(good) is Sidebar
    assert app_mod.APP_FACTORIES[good] is Sidebar, "the stand-in is replaced"
    assert app_mod.registered_factory(bad) is None
    assert isinstance(app_mod.APP_FACTORIES[bad], app_mod.LazyScreenFactory)
    assert "Could not import the screen registered for" in caplog.text


def test_an_app_name_is_still_registered_when_i18n_cannot_take_it(monkeypatch):
    """Registration must survive a translation seam that is not there.

    An app hands its translated names to the registry, which pushes them into
    `spacr.qt.i18n` if that module is loaded and exposes the hook. An
    untranslated sidebar row is a cosmetic loss; a raise here costs the row.
    """
    key = "__cov_wf_meta__"
    rows = ["Cov Widget"] * 9
    monkeypatch.setitem(app_mod.APP_META, key,
                        {"name": "Cov Widget", "translations": rows})
    taken = []
    monkeypatch.setattr(i18n_mod, "add_translation",
                        lambda source, values: taken.append((source, values)))
    app_mod._publish_meta(key)
    assert taken == [("Cov Widget", rows)]

    monkeypatch.setattr(i18n_mod, "add_translation", "not a function")
    app_mod._publish_meta(key)
    assert taken == [("Cov Widget", rows)], "a non-callable hook is skipped"


def test_the_dock_lists_every_app_even_when_no_icon_can_be_drawn(
        qtbot, qt_theme_applied, monkeypatch):
    """An icon set that cannot be read must not empty the app dock.

    Every row asks for its app's icon and skips the assignment when there is
    none, both when the column is built and when it is rebuilt after a theme
    change -- otherwise a machine whose icon resources fail to load gets rows
    with no way to navigate instead of text-only rows.
    """
    real_icon_for_app = app_mod._icon_for_app
    monkeypatch.setattr(app_mod, "_icon_for_app", lambda key: None)
    bar = Sidebar()
    qtbot.addWidget(bar)
    rows = [b for b in bar._items
            if b.property("navKey") not in (None, "", "__home__")]
    assert len(rows) > 5, "the dock still lists the registered apps"
    assert all(b.icon().isNull() for b in rows)
    bar.refresh_icons()
    assert all(b.icon().isNull() for b in rows), "still nothing to draw"

    monkeypatch.setattr(app_mod, "_icon_for_app", real_icon_for_app)
    bar.refresh_icons()
    assert any(not b.icon().isNull() for b in rows), (
        "an icon set that works must put icons back on the rows")


# --- construction ----------------------------------------------------------

def test_the_loading_screen_follows_the_preload_policy(qtbot, qt_theme_applied,
                                                       monkeypatch):
    """The cover comes down at once unless something is actually loading.

    Preloading is off by default (twenty seconds of torch to draw a window),
    so a window built under the lazy policy must close the cover it just put
    up rather than leave a grey sheet over the interface for good -- even
    when closing it raises. Under the eager policy the cover stays and is
    told the denominator, or its progress bar is a spinner with a number.
    """
    monkeypatch.setattr(app_mod._PipelinePreloader, "start", lambda self: None)
    monkeypatch.setattr(preferences, "get_preload_policy", lambda: "lazy")
    lazy_cover = _Recorder()
    monkeypatch.setattr(MainWindow, "_install_loading_screen",
                        lambda self: lazy_cover)
    lazy_window = MainWindow()
    qtbot.addWidget(lazy_window)
    assert lazy_window._preloader is None
    assert lazy_window._loading_screen is None
    assert lazy_cover.names() == ["close"]

    stuck_cover = _Recorder(close=RuntimeError("already deleted"))
    monkeypatch.setattr(MainWindow, "_install_loading_screen",
                        lambda self: stuck_cover)
    stuck_window = MainWindow()
    qtbot.addWidget(stuck_window)
    assert stuck_cover.names() == ["close"]
    assert stuck_window._loading_screen is None, "forgotten even when it threw"

    monkeypatch.setattr(preferences, "get_preload_policy", lambda: "eager")
    eager_cover = _Recorder()
    monkeypatch.setattr(MainWindow, "_install_loading_screen",
                        lambda self: eager_cover)
    eager_window = MainWindow()
    qtbot.addWidget(eager_window)
    eager_window._preloader._started = True   # nothing may import for real
    assert eager_window._loading_screen is eager_cover
    assert eager_window._preloader.total() > 0
    assert eager_cover.calls == [
        ("set_total", (eager_window._preloader.total(),))]


def test_a_window_that_cannot_tidy_its_tab_bars_still_opens(
        qtbot, qt_theme_applied, monkeypatch, caplog):
    """The window is worth more than the scroll arrows it wanted removed.

    Taking the arrows off every tab bar is a finishing touch applied during
    construction; a Qt build where that helper raises must still produce a
    window with its menus and its Home page, and say why in the log.
    """
    caplog.set_level(logging.DEBUG, logger="spacr.qt.app")

    def _refuse(_widget):
        raise RuntimeError("no tab bar here")

    monkeypatch.setattr(theme, "take_the_scroll_arrows_off", _refuse)
    window = MainWindow()
    qtbot.addWidget(window)
    assert window._stack.indexOf(window._startup) >= 0
    assert window.menuBar().actions(), "the menu bar was still built"
    assert "could not take the tab scroll arrows off" in caplog.text


def test_a_deleted_loading_screen_is_dropped_on_the_next_resize(win):
    """A resize must not fail because the cover it tracks is already gone.

    While the cover is up it is kept the size of the window. Qt can delete it
    underneath during teardown, and every later resize would then raise from
    a dead wrapper -- so the first failure forgets it.
    """
    event = QResizeEvent(win.size(), QSize(10, 10))
    live = _Recorder()
    win._loading_screen = live
    win.resizeEvent(event)
    assert live.calls == [("setGeometry", (win.rect(),))]

    dead = _Recorder(setGeometry=RuntimeError("already deleted"))
    win._loading_screen = dead
    win.resizeEvent(event)
    assert win._loading_screen is None
    assert dead.names() == ["setGeometry"]


def test_the_window_chrome_survives_missing_grips_and_a_missing_action(
        win, monkeypatch, caplog):
    """F11 must exist even when the menu that usually owns it did not.

    The chrome installer asks the glass module for drag-to-resize edges and
    reuses the Full screen action the Window menu made. A build without the
    resize helper still gets its corner buttons, and a window whose menu made
    no action makes its own -- or the button's tooltip advertises a dead key.
    """
    caplog.set_level(logging.DEBUG, logger="spacr.qt.app")
    from spacr.qt.widgets import glass

    def _refuse(_window):
        raise RuntimeError("no window handle")

    monkeypatch.setattr(glass, "let_the_user_resize", _refuse)
    del win._act_fullscreen
    win._install_fullscreen_button()
    assert "the window could not be made resizable" in caplog.text
    action = win._act_fullscreen
    assert action.shortcut().toString() == "F11"
    assert action in win.actions(), "the window itself must carry F11"
    assert win._fullscreen_button.objectName() == "FullScreenToggle"


# --- the menu bar is the title bar -----------------------------------------

def _mouse(kind, pos, button=Qt.MouseButton.LeftButton):
    point = QPointF(pos)
    return QMouseEvent(kind, point, point, button, button,
                       Qt.KeyboardModifier.NoModifier)


def test_double_clicking_the_menu_bar_maximises_the_window(win):
    """The window has no title bar, so the menu bar carries its gestures.

    Double-click to maximise and restore is what every desktop does with a
    title bar; without it this window can only be resized by its edges.
    """
    bar = win.menuBar()
    empty = QPoint(max(bar.width() - 4, 1), 2)
    assert bar.actionAt(empty) is None, "the gesture is only on empty bar"
    handled = win.eventFilter(
        bar, _mouse(QEvent.Type.MouseButtonDblClick, empty))
    assert handled is True, "the bar must consume the gesture"
    assert win.isMaximized() is True
    win.eventFilter(bar, _mouse(QEvent.Type.MouseButtonDblClick, empty))
    assert win.isMaximized() is False


def test_dragging_the_menu_bar_moves_the_window(win):
    """Without this the frameless window cannot be moved at all.

    A press on empty bar records the grab offset, movement carries the window
    with the pointer, and release ends the drag -- so a later stray move does
    not teleport a window nobody is dragging any more.
    """
    bar = win.menuBar()
    win.move(120, 90)
    empty = QPoint(max(bar.width() - 4, 1), 2)
    win.eventFilter(bar, _mouse(QEvent.Type.MouseButtonPress, empty))
    assert win._drag_from is not None
    start = win.frameGeometry().topLeft()
    win.eventFilter(bar, _mouse(QEvent.Type.MouseMove, empty + QPoint(40, 25),
                                Qt.MouseButton.NoButton))
    assert win.frameGeometry().topLeft() == start + QPoint(40, 25)

    win.eventFilter(bar, _mouse(QEvent.Type.MouseButtonRelease, empty))
    assert win._drag_from is None
    moved = win.frameGeometry().topLeft()
    win.eventFilter(bar, _mouse(QEvent.Type.MouseMove, empty + QPoint(400, 400),
                                Qt.MouseButton.NoButton))
    assert win.frameGeometry().topLeft() == moved, "the drag is over"


def test_a_window_state_change_relays_the_menu_bar(win, qtbot):
    """A menu opens where the BAR says its action is, not where it was.

    Going fullscreen resizes the bar and its corner widget in one step, and a
    menu opened before the layout has caught up is placed against the previous
    rectangle -- which is how pressing spaCR drops a menu under Help. The
    re-lay must also survive a bar with no corner widget.
    """
    win.show()
    qtbot.waitExposed(win)
    bar = win.menuBar()
    corner = bar.cornerWidget(Qt.Corner.TopRightCorner)
    assert corner is not None, "the window buttons live in the bar's corner"
    state = QWindowStateChangeEvent(Qt.WindowState.WindowNoState)
    corner.setGeometry(0, 0, 3, 3)
    win.changeEvent(state)
    assert corner.size() == corner.sizeHint(), (
        "the window buttons must be re-measured before a menu opens")

    QHBoxLayout(bar)                    # a bar that manages its own children
    bar.setCornerWidget(None, Qt.Corner.TopRightCorner)
    corner.setGeometry(0, 0, 3, 3)
    win.changeEvent(state)
    assert corner.size() == QSize(3, 3), "no corner widget, nothing to measure"
    assert bar.layout() is not None and len(bar.actions()) > 1


def test_menu_roles_cannot_be_pinned_without_a_menu_bar(win, monkeypatch):
    """macOS moves actions it thinks it recognises out of the menu entirely.

    Every action is given an explicit role so Qt's text heuristic cannot
    relocate "Options" or "Setup". The walk runs during teardown too, when
    the bar may already be gone, and must then find nothing rather than raise.
    """
    actions = win._menu_bar_actions()
    assert len(actions) > 5
    assert any(a.text() for a in actions)
    monkeypatch.setattr(win, "menuBar", lambda: None)
    assert win._menu_bar_actions() == []


# --- About, theme refresh, Home, shutdown ----------------------------------

def test_about_still_opens_when_the_logo_cannot_be_loaded(win, monkeypatch,
                                                          tmp_path):
    """The panel states the version; the mark is decoration on top of that.

    An install with an incomplete resource folder must still get a readable
    About panel: setting a null pixmap instead of skipping it would blank the
    label's size hint and leave a hole above the name.
    """
    shown = []
    monkeypatch.setattr(QDialog, "exec",
                        lambda dialog, *a, **k: shown.append(dialog))
    win._show_about()
    assert not shown[0].findChildren(QLabel)[0].pixmap().isNull()

    monkeypatch.setattr(iconset, "RESOURCE_DIR", str(tmp_path))
    win._show_about()
    second = shown[1]
    assert second.findChildren(QLabel)[0].pixmap().isNull()
    body = "\n".join(w.text() for w in second.findChildren(QLabel))
    assert "Version" in body and "Olafsson Lab" in body


def test_a_theme_refresh_skips_screens_that_cannot_hide_immature_apps(win):
    """One screen without the hook must not stop the others being refreshed.

    Maturity visibility is re-applied to every open screen when the
    preference changes, and screens come from many modules -- so the refresh
    asks before it calls, and a screen that has the hook still gets it.
    """
    asked = _Recorder()
    plain = object()
    win._screens = {"with-hook": asked, "without-hook": plain}
    win.refresh_theme()
    assert asked.names() == ["refresh_maturity_visibility"]
    assert not hasattr(plain, "refresh_maturity_visibility")


def test_home_can_be_rebuilt_when_there_is_no_page_to_replace(win):
    """A font-scale change rebuilds Home; the first build has none to drop.

    Rebuilding normally removes the outgoing page from the stack, so the
    count is unchanged. Called with no page -- or after one was already
    removed -- it must install one instead of removing what is not there.
    """
    before = win._stack.count()
    old = win._startup
    win._rebuild_startup_page()
    assert win._stack.count() == before, "the old page was taken out"
    assert win._startup is not old
    assert win._stack.indexOf(old) == -1

    win._startup = None
    win._rebuild_startup_page()
    assert win._stack.count() == before + 1, "nothing was there to remove"
    assert win._stack.indexOf(win._startup) >= 0


def test_closing_the_window_quits_the_application_exactly_once(win,
                                                               monkeypatch):
    """With quitOnLastWindowClosed off, this close is what ends the program.

    A figure window closing must not take the session down, so nothing else
    may quit -- and this window must, or spaCR keeps running with nothing on
    screen. During teardown there is no application left, and asking one to
    quit anyway would raise from inside a close handler.
    """
    quits = _Recorder()
    holder = {"instance": quits}

    class _StubQApplication:
        @staticmethod
        def instance():
            return holder["instance"]

    monkeypatch.setattr(app_mod, "QApplication", _StubQApplication)
    first = QCloseEvent()
    win.closeEvent(first)
    assert first.isAccepted() is True
    assert quits.names() == ["quit"]

    holder["instance"] = None
    win.closeEvent(QCloseEvent())
    assert quits.names() == ["quit"], "no application, nothing to quit"


# --- the dock, the drawer and the backdrop ---------------------------------

def test_the_dock_can_be_moved_before_its_menu_action_exists(win):
    """The dock preference is applied during construction, menu or no menu.

    "All apps" is greyed out with an explanation when the dock is hidden,
    because a control that silently does nothing is worse than one that says
    why. The placement itself runs once before the menu is finished.
    """
    win.apply_dock_mode("hidden")
    assert win._act_all_apps.isEnabled() is False
    assert "Preferences" in win._act_all_apps.toolTip()

    del win._act_all_apps
    win.apply_dock_mode("locked")
    assert win._dock_mode == "locked"
    assert win._sidebar.parent() is win._dock_slot


def test_the_backdrop_can_be_blanked_without_its_menu_item(win):
    """Ctrl+B and the menu tick have to agree, when there is a tick.

    Blanking stops each backdrop before hiding it -- one still rendering
    spends the threads and shows nothing for them -- and keeps the menu item
    in step. The shortcut is installed before the menu, so it must also work
    with no item to update.
    """
    class _Backdrop(QWidget):
        def __init__(self, parent):
            super().__init__(parent)
            self.states = []

        def set_animating(self, on):
            self.states.append(on)

    backdrop = _Backdrop(win)
    win._act_backdrop.setChecked(True)
    assert win._set_backdrop_blank(True) >= 1
    assert backdrop.states[-1] is False
    assert backdrop.isHidden() is True
    assert win._act_backdrop.isChecked() is False

    del win._act_backdrop
    assert win._set_backdrop_blank(False) >= 1
    assert backdrop.states[-1] is True
    assert backdrop.isHidden() is False


def test_the_drawer_shortcuts_do_nothing_when_there_is_no_drawer(win):
    """Ctrl+Shift+A and a drawer click must not raise on a dockless window.

    The drawer is the only pointer-driven way to reach another module from
    inside one, so the shortcut that opens it and the click that closes it
    run on every window -- including one whose drawer could not be built.
    """
    win._dock_mode = "auto"
    drawer = _Recorder()
    win._app_drawer = drawer
    win.toggle_app_drawer()
    win._on_drawer_navigated("mask")
    assert drawer.names() == ["toggle", "close"]

    win._app_drawer = None
    win.toggle_app_drawer()
    win._on_drawer_navigated("mask")
    assert drawer.names() == ["toggle", "close"], "no drawer, no calls"


def test_a_screensaver_that_will_not_open_is_not_held(win, monkeypatch):
    """The window holds the saver, or Python frees it as it appears.

    The full-screen background is a separate window with no other reference,
    so it is kept here and dropped when destroyed. A build that cannot make
    one must report that instead of storing ``None`` and claiming success.
    """
    from spacr.qt import screensaver

    saver = QWidget()
    monkeypatch.setattr(screensaver, "show_screensaver", lambda parent: saver)
    assert win._show_the_screensaver() is True
    assert win._screensaver is saver

    win._screensaver = None
    monkeypatch.setattr(screensaver, "show_screensaver", lambda parent: None)
    assert win._show_the_screensaver() is False
    assert win._screensaver is None


def test_the_backdrop_only_reports_a_zoom_rate_that_changed(win, monkeypatch):
    """The status line is the only feedback the zoom keys have.

    When there is no fractal to steer the nudge returns nothing, and the
    window must leave the key alone so the arrow still moves the selection in
    whatever table has focus.
    """
    monkeypatch.setattr(fractal_travel, "nudge_zoom_rate", lambda steps: 0.0)
    win.statusBar().clearMessage()
    assert win._steer_the_backdrop(1) is False
    assert win.statusBar().currentMessage() == ""

    monkeypatch.setattr(fractal_travel, "nudge_zoom_rate", lambda steps: 2.5)
    assert win._steer_the_backdrop(1) is True
    assert "2.50" in win.statusBar().currentMessage()


def test_the_arrows_and_the_wheel_steer_the_backdrop(win, monkeypatch):
    """Up, Down and the wheel are the spaceout zoom controls.

    Each has to pass the direction it stands for -- an inverted wheel or a
    Down key that zooms in is the kind of thing only a test notices -- and
    the window claims the event only when the backdrop took it.
    """
    steps = []

    def _nudge(count):
        steps.append(count)
        return 3.0

    monkeypatch.setattr(fractal_travel, "nudge_zoom_rate", _nudge)
    win.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Up,
                                Qt.KeyboardModifier.NoModifier))
    win.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Down,
                                Qt.KeyboardModifier.NoModifier))
    win.wheelEvent(QWheelEvent(
        QPointF(5.0, 5.0), QPointF(5.0, 5.0), QPoint(0, 0), QPoint(0, 240),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False))
    assert steps == [1, -1, 2]
    assert "3.00" in win.statusBar().currentMessage()


# --- building and rebuilding screens ---------------------------------------

def test_a_screen_can_be_rebuilt_before_it_was_ever_built(win, monkeypatch):
    """A committed setting rebuilds the form; the first open has no old form.

    The replacement is built before the old page leaves the stack so the
    window never flashes Home in between. With no old page there is nothing
    to close, and the new one still has to end up on screen.
    """
    from spacr.qt.screens.app_screen import AppScreen

    fresh = QWidget()
    monkeypatch.setattr(win, "_build_screen", lambda key: fresh)
    before = AppScreen.values_the_next_screen_is_built_for
    assert "mask" not in win._screens
    win.rebuild_app_screen("mask", {"nucleus_channel": 1})
    assert win._screens["mask"] is fresh
    assert win._stack.currentWidget() is fresh
    assert AppScreen.values_the_next_screen_is_built_for is before


def test_the_preparing_card_can_be_taken_down_twice(win):
    """The card goes up before a slow build and comes down after it, always.

    The build can fail before the card was created, so the teardown runs with
    nothing to take down. If that raised, one screen failing to build would
    leave "Preparing…" on the window for the rest of the session.
    """
    card = QWidget(win)
    card.show()
    win._hide_preparing(card)
    assert card.isHidden() is True
    assert win._hide_preparing(None) is None
    assert card.isHidden() is True


def test_a_plugin_screen_must_be_a_widget(win, monkeypatch):
    """A plugin that returns the wrong thing has to say so, loudly.

    A contributed factory's result goes straight into the window's stack, and
    adding a non-widget there is a C++ type error with no useful message --
    so the type is checked and the failure names the factory that caused it.
    """
    from spacr import plugins

    class _Contribution:
        key = "__cov_wf_plugin__"
        screen_factory = "cov_wf_plugin:make_screen"

    monkeypatch.setattr(plugins, "get_app", lambda key: _Contribution())
    made = QWidget()
    monkeypatch.setattr(plugins, "load_object",
                        lambda path: (lambda app_key: made))
    assert win._build_screen("__cov_wf_plugin__") is made

    monkeypatch.setattr(plugins, "load_object",
                        lambda path: (lambda app_key: "a string"))
    with pytest.raises(TypeError) as raised:
        win._build_screen("__cov_wf_plugin__")
    assert "cov_wf_plugin:make_screen" in str(raised.value)
    assert "expected QWidget" in str(raised.value)
