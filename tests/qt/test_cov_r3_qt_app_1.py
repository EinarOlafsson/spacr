"""What ``spacr.qt.app`` does when a part of the registry or the window fails.

The import-time registry comes first -- an undeclared app module, a
module whose ``register()`` raises, and the plugin loop. Those branches
run once, while the module is imported, so they are driven by executing
the same file again in a private namespace with only the broken
collaborator replaced; the registry that copy builds is its own.

Then the window: a menu bar that will not re-lay, a broken logo path, a
Home page that will not close, an app screen that vanished mid-shutdown,
and the backdrop hotkeys with the fractal module gone. Each is a branch a
user meets only where something is already wrong, which is when a crash
is least affordable.
"""
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import logging
import platform
import sys
import types

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QWindowStateChangeEvent
from PySide6.QtWidgets import QDialog, QLabel, QWidget

from spacr import plugins as plugins_mod
from spacr.qt import app as app_mod
from spacr.qt import app_catalog, iconset
from spacr.qt import screensaver as screensaver_mod
from spacr.qt.app import MainWindow
from spacr.qt.screens import app_screen as app_screen_mod
from spacr.qt.widgets import fractal_travel

#: A declared module whose row is metadata only, used as the stand-in for
#: "an app module that has not been declared yet".
_DECLARED_MODULE = "spacr.qt.screens.tabulate"


@pytest.fixture
def win(qtbot, qt_theme_applied):
    """A live MainWindow, cleaned up by pytest-qt."""
    window = MainWindow()
    qtbot.addWidget(window)
    return window


class _Log(logging.Handler):
    """Collect what ``spacr.qt.app`` logged, independent of propagation."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())

    def text(self):
        return "\n".join(self.messages)


@pytest.fixture
def app_log():
    """Records attached straight to the module's own logger.

    These ledges report at DEBUG, which the session's logging setup may be
    filtering out, so the level is opened here and put back afterwards.
    """
    handler = _Log()
    logger = logging.getLogger("spacr.qt.app")
    level, disabled = logger.level, logging.root.manager.disable
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logging.disable(logging.NOTSET)
    try:
        yield handler
    finally:
        logging.disable(disabled)
        logger.setLevel(level)
        logger.removeHandler(handler)


@contextlib.contextmanager
def _unreachable_children(window):
    """Make ``findChildren`` raise inside the block, and only there.

    Restored by hand: pytest-qt closes the window during teardown, and
    ``closeEvent`` walks the children too.
    """
    window.findChildren = _boom
    try:
        yield
    finally:
        del window.findChildren


@pytest.fixture
def side_tables_restored():
    """Undo the metadata a re-imported copy pushes into the side tables.

    ``register_app`` fans an app's title and intro out into
    ``app_screen.APP_TITLES`` and friends -- globals shared with the live
    registry -- and a fake plugin must not outlive its test.
    """
    saved = []
    for module_name, attribute, _field in app_mod._META_TARGETS:
        module = sys.modules.get(module_name)
        table = getattr(module, attribute, None)
        if isinstance(table, dict):
            saved.append((table, dict(table)))
    yield
    for table, snapshot in saved:
        table.clear()
        table.update(snapshot)


def _fresh_app_module():
    """Execute ``spacr/qt/app.py`` again under its own name, privately.

    Never put in ``sys.modules``: importers keep the module the session
    already has, and only the import-time registry work is re-run.
    """
    spec = importlib.util.spec_from_file_location(
        "spacr.qt.app", app_mod.__file__)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    return fresh


class _Contribution:
    """The shape ``spacr.plugins.plugin_apps`` hands the registry."""

    def __init__(self, key, *, section="core", stage="beta", name=None):
        self.key = key
        self.name = name if name is not None else f"Plugin {key}"
        self.description = f"Contributed app {key}"
        self.section = section
        self.stage = stage


def _keys(module):
    return [row[0] for row in module.APPS]


def _boom(*_args, **_kwargs):
    raise RuntimeError("this collaborator is broken")


# --- the import-time registry ----------------------------------------------

def test_an_app_module_with_no_catalog_row_is_imported_and_registers_itself(
        monkeypatch, side_tables_restored):
    """A screen that predates the catalog must still get its sidebar row.

    Declared apps register from strings so their pandas/torch imports stay
    off the startup path. A module that declares nothing takes the older
    contract -- import it, call its ``register()`` -- and dropping that
    fallback would make the app vanish from the sidebar with no warning.
    """
    called = []
    stub = types.ModuleType(_DECLARED_MODULE)
    stub.register = lambda: called.append(_DECLARED_MODULE)
    real_import = importlib.import_module

    def _import(name, package=None):
        if name == _DECLARED_MODULE:
            return stub
        return real_import(name) if package is None else real_import(
            name, package)

    monkeypatch.setattr(importlib, "import_module", _import)

    _fresh_app_module()
    assert called == [], "a declared module is registered from its row alone"

    real_declared = app_catalog.declared_for
    monkeypatch.setattr(
        app_catalog, "declared_for",
        lambda module: None if module == _DECLARED_MODULE
        else real_declared(module))
    fresh = _fresh_app_module()
    assert called == [_DECLARED_MODULE], (
        "an undeclared module must be imported and asked to register")
    assert len(_keys(fresh)) > 20, "the rest of the registry still filled"


def test_a_screen_that_raises_while_registering_costs_only_that_screen(
        monkeypatch, app_log, side_tables_restored):
    """One module's import-time bug must not stop the window from opening.

    An exception escaping this loop leaves the sidebar half-built and no
    window at all. The failure is logged with the module's name so a user
    can say which app is broken.
    """
    stub = types.ModuleType(_DECLARED_MODULE)
    stub.register = _boom
    real_import = importlib.import_module
    real_declared = app_catalog.declared_for
    monkeypatch.setattr(
        importlib, "import_module",
        lambda name, package=None: stub if name == _DECLARED_MODULE
        else real_import(name))
    monkeypatch.setattr(
        app_catalog, "declared_for",
        lambda module: None if module == _DECLARED_MODULE
        else real_declared(module))

    fresh = _fresh_app_module()
    assert _DECLARED_MODULE in app_log.text(), app_log.text()
    assert "Could not register" in app_log.text()
    assert len(_keys(fresh)) > 20, "every other app still registered"
    assert hasattr(fresh, "MainWindow"), "the module finished importing"


def test_a_plugin_claiming_a_built_in_key_is_dropped_not_swapped_in(
        monkeypatch, side_tables_restored):
    """A contribution may add an app; it may never replace one.

    Letting a plugin take ``mask``'s key would repoint a built-in
    pipeline's Run button at third-party code -- and the older bug kept
    both rows, so the sidebar listed the app twice. The collision is
    recorded so the plugin author is told why their app is missing.
    """
    builtin = app_mod._BUILTIN_APPS[0][0]
    diagnostics = []
    monkeypatch.setattr(plugins_mod, "record_diagnostic",
                        lambda key, message, *a, **k:
                        diagnostics.append((key, message)))
    monkeypatch.setattr(
        plugins_mod, "plugin_apps",
        lambda: (_Contribution(builtin), _Contribution("cov_r3_plugin_ok")))

    fresh = _fresh_app_module()
    keys = _keys(fresh)
    assert keys.count(builtin) == 1, "the built-in row was kept, and once"
    assert dict((row[0], row[1]) for row in fresh.APPS)[builtin] != (
        f"Plugin {builtin}"), "the built-in name survived the collision"
    assert "cov_r3_plugin_ok" in keys, "a plugin with its own key still lands"
    assert [key for key, _ in diagnostics] == [builtin]
    assert "collides" in diagnostics[0][1]


def test_a_plugin_the_registry_refuses_is_reported_and_the_others_load(
        monkeypatch, side_tables_restored):
    """One malformed contribution must not cost every other plugin.

    ``register_app`` validates what ``spacr.plugins`` does not -- here an
    unknown maturity stage. Without the per-plugin catch, the first bad
    contribution takes every plugin behind it down, silently.
    """
    diagnostics = []
    monkeypatch.setattr(plugins_mod, "record_diagnostic",
                        lambda key, message, *a, **k:
                        diagnostics.append((key, message)))
    monkeypatch.setattr(
        plugins_mod, "plugin_apps",
        lambda: (_Contribution("cov_r3_plugin_bad", stage="mythical"),
                 _Contribution("cov_r3_plugin_after")))

    fresh = _fresh_app_module()
    keys = _keys(fresh)
    assert "cov_r3_plugin_bad" not in keys, "an invalid stage is not registered"
    assert "cov_r3_plugin_after" in keys, "the plugin behind it still loaded"
    assert [key for key, _ in diagnostics] == ["cov_r3_plugin_bad"]
    assert "was not registered" in diagnostics[0][1]
    assert "mythical" in diagnostics[0][1], "the reason names the bad value"


def test_a_plugin_source_that_explodes_leaves_the_built_ins_intact(
        monkeypatch, app_log, side_tables_restored):
    """Plugin discovery is third-party code; it must not own the launch.

    An unreadable entry point or a broken plugin package makes
    ``plugin_apps()`` raise, and spaCR still has to start with its
    built-ins: the alternative is an install where no window opens at all.
    """
    monkeypatch.setattr(plugins_mod, "plugin_apps", _boom)

    fresh = _fresh_app_module()
    assert "Could not add plugin apps" in app_log.text(), app_log.text()
    keys = _keys(fresh)
    assert app_mod._BUILTIN_APPS[0][0] in keys
    assert len(keys) == len(app_mod._BUILTIN_APPS)


# --- the window's decorations ----------------------------------------------

def test_a_menu_bar_that_cannot_be_re_laid_does_not_break_full_screen(
        win, qtbot, app_log, monkeypatch):
    """Going full screen must not raise out of an event handler.

    The re-lay is what makes a menu open under its own title after the bar
    resizes -- a nicety. A window whose menu bar is already torn down still
    has to process the state change, so the failure is logged and dropped.
    """
    win.show()
    qtbot.waitExposed(win)
    bar = win.menuBar()
    state = QWindowStateChangeEvent(Qt.WindowState.WindowNoState)
    corner = bar.cornerWidget(Qt.Corner.TopRightCorner)
    corner.setGeometry(0, 0, 3, 3)
    win.changeEvent(state)
    assert corner.size() == corner.sizeHint(), "the healthy bar re-measured"
    assert "could not re-lay" not in app_log.text()

    monkeypatch.setattr(win, "menuBar", _boom)
    win.changeEvent(state)
    # TWO FAILURES, TWO MESSAGES. The re-lay was split when the ghosting
    # fix landed: asking for the bar and re-laying it can each fail on
    # their own, and one message for both sent the reader to the wrong
    # half. This is the first -- the bar could not be asked for at all.
    assert "the menu bar could not be asked for" in app_log.text()
    assert len(bar.actions()) > 1, "the bar itself is untouched"


def test_about_still_names_the_version_when_the_build_line_is_unreadable(
        win, monkeypatch):
    """The version is the point of the panel; the build line is a footnote.

    "Python 3.x . Qt 6.y" comes from two libraries that can both fail on an
    odd install, and losing that line must not lose the dialog: a user
    filing a bug is reading the version off it.
    """
    shown = []
    monkeypatch.setattr(QDialog, "exec",
                        lambda dialog, *a, **k: shown.append(dialog))
    win._show_about()
    healthy = "\n".join(w.text() for w in shown[0].findChildren(QLabel))
    assert "Python" in healthy, healthy

    monkeypatch.setattr(platform, "python_version", _boom)
    win._show_about()
    body = "\n".join(w.text() for w in shown[1].findChildren(QLabel))
    assert "Python" not in body, body
    assert "Version" in body and "Olafsson Lab" in body


def test_about_opens_when_the_logo_path_itself_cannot_be_built(
        win, monkeypatch):
    """A broken resource directory costs the mark, not the panel.

    A missing logo file is already handled; a ``RESOURCE_DIR`` that is not
    a path at all raises before there is a pixmap to test, and the panel
    must still show the version and the licence.
    """
    shown = []
    monkeypatch.setattr(QDialog, "exec",
                        lambda dialog, *a, **k: shown.append(dialog))
    monkeypatch.setattr(iconset, "RESOURCE_DIR", 17)
    win._show_about()
    mark = shown[0].findChildren(QLabel)[0]
    assert mark.pixmap().isNull(), "no mark could be loaded"
    body = "\n".join(w.text() for w in shown[0].findChildren(QLabel))
    assert "Version" in body and "BSD 3-Clause" in body


def test_home_is_rebuilt_even_when_the_outgoing_page_will_not_close(
        win, monkeypatch):
    """A font-scale change must always leave exactly one Home page.

    The old page is closed before deletion so it drops its run-registry
    subscription now. One that refuses to close is still removed and
    replaced -- otherwise a broken Home freezes the window on itself.
    """
    old = win._startup
    monkeypatch.setattr(old, "close", _boom)
    before = win._stack.count()

    win._rebuild_startup_page()
    assert win._startup is not old, "a new Home page was installed"
    assert win._stack.indexOf(old) == -1, "the old page left the stack"
    assert win._stack.count() == before


# --- shutdown ---------------------------------------------------------------

class _StubScreen:
    """An owned app screen, in whatever state the test needs it."""

    def __init__(self, name, outcome=True):
        self.name = name
        self.outcome = outcome
        self.closed = []

    def close(self):
        self.closed.append(self.name)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def test_a_screen_deleted_under_us_does_not_stop_the_shutdown(
        win, monkeypatch):
    """A screen Qt already destroyed has nothing left to drain.

    Closing asks every owned screen to shut down first, because a parent's
    close is not delivered to its children. A RuntimeError means a dead C++
    object: the screens behind it must still be asked, and the window must
    still close.
    """
    gone = _StubScreen("gone", RuntimeError("already deleted"))
    live = _StubScreen("live")
    monkeypatch.setattr(app_screen_mod, "AppScreen", _StubScreen)
    monkeypatch.setattr(win, "_screens", {"a": gone, "b": live})

    class _NoApplication:
        @staticmethod
        def instance():
            return None

    monkeypatch.setattr(app_mod, "QApplication", _NoApplication)
    event = QCloseEvent()
    win.closeEvent(event)
    assert live.closed == ["live"], "the screen behind it was still asked"
    assert event.isAccepted() is True
    assert win._closing is True


def test_a_screen_that_raises_while_closing_keeps_the_window_open(
        win, app_log, monkeypatch):
    """An unexpected error while draining must not destroy live workers.

    RuntimeError means the screen is already gone; anything else means the
    drain did not finish, and closing anyway tears down a running analysis
    mid-write. The close is refused and the reason logged.
    """
    angry = _StubScreen("angry", ValueError("mid-write"))
    later = _StubScreen("later")
    monkeypatch.setattr(app_screen_mod, "AppScreen", _StubScreen)
    monkeypatch.setattr(win, "_screens", {"a": angry, "b": later})

    event = QCloseEvent()
    win.closeEvent(event)
    assert event.isAccepted() is False, "the window stays up"
    assert win._closing is False, "a refused close must be retryable"
    assert angry.closed == ["angry"], "the failing screen was asked"
    assert later.closed == [], "the shutdown stopped at the failure"
    assert "Could not close an owned application screen" in app_log.text()


# --- the backdrop hotkeys ---------------------------------------------------

class _Backdrop(QWidget):
    """A decoration that answers ``set_animating``, like the real ones."""

    def __init__(self, parent):
        super().__init__(parent)
        self.states = []

    def set_animating(self, on):
        self.states.append(bool(on))


def test_blanking_the_backdrops_never_raises_whatever_refuses(
        win, app_log, monkeypatch):
    """Ctrl+B is a decoration switch, not a feature: it must never raise.

    The blank is the user's request and the menu tick is bookkeeping, so a
    destroyed action must not undo the blanking that did happen -- and a
    window whose children cannot be walked at all answers zero rather than
    taking the keystroke, and the menu it came from, down with it.
    """
    backdrop = _Backdrop(win)
    assert win._set_backdrop_blank(True) >= 1
    assert backdrop.states and all(state is False for state in backdrop.states)
    assert backdrop.isVisible() is False

    class _RefusingAction:
        def setChecked(self, _value):
            raise RuntimeError("this action is gone")

    monkeypatch.setattr(win, "_act_backdrop", _RefusingAction())
    backdrop.states.clear()
    assert win._set_backdrop_blank(True) >= 1, "the backdrop was still hidden"
    assert backdrop.states == [False], "the tick refused, the blank happened"

    with _unreachable_children(win):
        assert win._set_backdrop_blank(False) == 0, "nothing was reached"
    assert "could not reach the backdrops" in app_log.text()


def test_toggling_the_animation_survives_a_window_it_cannot_walk(
        win, app_log):
    """Ctrl+T answers a number so the menu can report what it reached.

    Both kinds of backdrop -- ambient engines and the spaceout fractal --
    answer ``set_animating``. A window that cannot be walked answers zero;
    a decoration must not be able to break a menu.
    """
    backdrop = _Backdrop(win)
    assert win._set_backdrop_animating(True) >= 1
    assert backdrop.states == [True]

    with _unreachable_children(win):
        assert win._set_backdrop_animating(False) == 0
    assert backdrop.states == [True], "nothing was reached the second time"
    assert "could not reach the backdrops" in app_log.text()


def test_a_screensaver_that_will_not_open_says_so_and_reports_false(
        win, qtbot, app_log, monkeypatch):
    """The caller needs to know whether a window actually opened.

    The full-screen backdrop is a separate window precisely so nothing has
    to be restored when it fails. Returning False rather than raising is
    what keeps the menu item usable on a build with no screensaver.
    """
    saver = QWidget()
    qtbot.addWidget(saver)
    monkeypatch.setattr(screensaver_mod, "show_screensaver",
                        lambda parent: saver)
    assert win._show_the_screensaver() is True
    assert win._screensaver is saver, "the only reference is held"

    monkeypatch.setattr(screensaver_mod, "show_screensaver", _boom)
    assert win._show_the_screensaver() is False
    assert "could not open the full-screen background" in app_log.text()


def test_restarting_a_backdrop_that_cannot_be_restarted_reports_false(
        win, monkeypatch):
    """Ctrl+R and the menu entry come here, so they cannot drift apart.

    The status message is shown only when a backdrop really did go back to
    the surface: a fractal module that raises must report no restart rather
    than announce one that did not happen.
    """
    calls = []
    monkeypatch.setattr(fractal_travel, "_LIVE_CONTROLS", [object()])
    monkeypatch.setattr(fractal_travel, "restart_the_dive",
                        lambda: calls.append("restart"))
    win.statusBar().clearMessage()
    assert win._restart_the_backdrop() is True
    assert calls == ["restart"]
    assert win.statusBar().currentMessage() != ""

    win.statusBar().clearMessage()
    monkeypatch.setattr(fractal_travel, "restart_the_dive", _boom)
    assert win._restart_the_backdrop() is False
    assert win.statusBar().currentMessage() == "", "nothing was announced"


def test_steering_a_backdrop_that_is_not_there_reports_false(
        win, monkeypatch):
    """Up and Down must fall through to whatever else wants the key.

    The window handles the arrows only when a backdrop took the change; if
    the fractal module raises, the keystroke has to be left unaccepted so a
    list or a table still moves its selection.
    """
    monkeypatch.setattr(fractal_travel, "nudge_zoom_rate", lambda steps: 1.5)
    win.statusBar().clearMessage()
    assert win._steer_the_backdrop(1) is True
    assert "1.50" in win.statusBar().currentMessage()

    win.statusBar().clearMessage()
    monkeypatch.setattr(fractal_travel, "nudge_zoom_rate", _boom)
    assert win._steer_the_backdrop(1) is False
    assert win.statusBar().currentMessage() == "", "no rate was announced"
