"""``MainWindow`` around its screens: construction, About, Preferences, theme,
Home rebuilds, window state, the Edit menu and closing.

Every test drives the real window offscreen; only a collaborator that is
meant to be broken, and the modal ``QDialog.exec`` that would block the run,
are replaced.  Failure arms sit beside the working arm they are measured
against.
"""
from __future__ import annotations

import logging
import platform
import sys
import types

import pytest
from PySide6.QtCore import QEvent, QPointF, QSize, Qt
from PySide6.QtGui import QCloseEvent, QMouseEvent, QWindowStateChangeEvent
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QPlainTextEdit,
                               QTextEdit, QVBoxLayout, QWidget)

from spacr.qt import app as app_mod
from spacr.qt import iconset, theme
from spacr.qt import preferences as preferences_mod
from spacr.qt.app import MainWindow, _DragsTheWindowByTheMenuBar
from tests.qt.test_cov_qt_app import _close_owned_screens, win  # noqa: F401
from tests.qt.test_main_window_screens import app_log  # noqa: F401

pytestmark = pytest.mark.qt


def _boom(*_args, **_kwargs):
    raise RuntimeError("this collaborator is broken")


class _Recorder:
    """Answers any call, remembers it; ``_Recorder(close=exc)`` raises exc."""

    def __init__(self, **behaviour):
        self.calls: list = []
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


class _StubScreen(QWidget):
    """A cached screen with a controllable close result."""

    def __init__(self, name, outcome=True, parent=None):
        super().__init__(parent)
        self.name = name
        self.outcome = outcome
        self.closed: list = []

    def close(self):
        self.closed.append(self.name)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class _RefusedClose(QCloseEvent):
    """A close event that stays un-accepted however it is handled."""

    def isAccepted(self):
        return False


def _window(qtbot, **kwargs):
    window = MainWindow(**kwargs)
    qtbot.addWidget(window, before_close_func=_close_owned_screens)
    return window


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def test_a_window_is_built_whatever_its_finishing_touches_do(
        qtbot, qt_theme_applied, monkeypatch, caplog, subtests):
    """Module hints, tab-bar tidying and the loading cover are optional; the
    shortcuts and the menu-bar drag filter are installed on every window."""
    import spacr.qt.module_hints as module_hints

    with subtests.test(msg="shortcuts bound exactly once each"):
        from PySide6.QtGui import QAction, QShortcut

        from spacr.qt import shortcuts as _sc
        window = _window(qtbot)
        shortcut_keys = [s.key().toString()
                         for s in window.findChildren(QShortcut)]
        bound = set(shortcut_keys) | {seq.toString()
                                      for act in window.findChildren(QAction)
                                      for seq in act.shortcuts()}
        assert [k for k in ("Ctrl+0", "Ctrl+K", "Ctrl+1", "Ctrl+9",
                            "Ctrl+Shift+H", "F1") if k not in bound] == []
        assert [k for k in _sc.BOUND_ELSEWHERE if k in shortcut_keys] == []

    with subtests.test(msg="the drag filter sits on the real menu bar"):
        assert isinstance(window._menu_drag, _DragsTheWindowByTheMenuBar)
        assert window._menu_drag.parent() is window

    with subtests.test(msg="module hints that will not install"), \
            monkeypatch.context() as m:
        m.setattr(module_hints, "install_module_hints", _boom)
        window = _window(qtbot)
        assert window._module_hints is None
        assert window._stack.indexOf(window._startup) >= 0

    with subtests.test(msg="tab bars that cannot be tidied"), \
            monkeypatch.context() as m:
        caplog.set_level(logging.DEBUG, logger="spacr.qt.app")
        m.setattr(theme, "take_the_scroll_arrows_off", _boom)
        window = _window(qtbot)
        assert window._stack.indexOf(window._startup) >= 0
        assert window.menuBar().actions()
        assert "could not take the tab scroll arrows off" in caplog.text

    with subtests.test(msg="the loading cover follows the preload policy"), \
            monkeypatch.context() as m:
        m.setattr(app_mod._PipelinePreloader, "start", lambda self: None)
        m.setattr(preferences_mod, "get_preload_policy", lambda: "lazy")
        lazy_cover = _Recorder()
        m.setattr(MainWindow, "_install_loading_screen", lambda self: lazy_cover)
        lazy = _window(qtbot)
        assert (lazy._preloader, lazy._loading_screen) == (None, None)
        assert lazy_cover.names() == ["close"]

        stuck_cover = _Recorder(close=RuntimeError("already deleted"))
        m.setattr(MainWindow, "_install_loading_screen",
                  lambda self: stuck_cover)
        stuck = _window(qtbot)
        assert stuck_cover.names() == ["close"]
        assert stuck._loading_screen is None

        m.setattr(preferences_mod, "get_preload_policy", lambda: "eager")
        eager_cover = _Recorder()
        m.setattr(MainWindow, "_install_loading_screen",
                  lambda self: eager_cover)
        eager = _window(qtbot)
        eager._preloader._started = True
        assert eager._loading_screen is eager_cover
        assert eager._preloader.total() == len(app_mod._PipelinePreloader._MODULES)
        assert eager_cover.calls == [("set_total", (eager._preloader.total(),))]


# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------

def test_the_about_panel_states_version_build_and_licence_whatever_breaks(
        win, monkeypatch, tmp_path, subtests):
    """Mark, name, version, build line, licence and lab; an unreadable version
    says "unknown", a broken build line or logo costs only itself."""
    import spacr
    from PySide6.QtCore import qVersion

    from spacr.qt.terms import LICENSE_NAME

    shown: list = []
    monkeypatch.setattr(QDialog, "exec",
                        lambda dialog, *a, **k: shown.append(dialog))

    def _about():
        win._show_about()
        dialog = shown[-1]
        labels = dialog.findChildren(QLabel)
        return dialog, labels[0], [w.text() for w in labels[1:]]

    with subtests.test(msg="healthy"):
        dialog, mark, texts = _about()
        assert (dialog.objectName(), dialog.windowTitle()) == (
            "AboutDialog", "About spaCR")
        assert not mark.pixmap().isNull()
        assert texts[0] == "spaCR"
        assert texts[2] == f"Version {spacr.__version__}"
        assert texts[3] == f"Python {platform.python_version()} · Qt {qVersion()}"
        assert LICENSE_NAME in texts[4]
        assert texts[5] == "© Olafsson Lab"

    cases = [
        ("unreadable version", (spacr, "__version__", None), "Version unknown",
         True),
        ("unreadable build line", (platform, "python_version", _boom),
         f"Version {spacr.__version__}", False),
        ("logo missing", (iconset, "RESOURCE_DIR", str(tmp_path)),
         f"Version {spacr.__version__}", True),
        ("logo path not a path", (iconset, "RESOURCE_DIR", 17),
         f"Version {spacr.__version__}", True),
    ]
    for name, (target, attr, value), version_line, has_build in cases:
        with subtests.test(msg=name), monkeypatch.context() as m:
            if value is None:
                m.delattr(target, attr)
            else:
                m.setattr(target, attr, value)
            _dialog, mark, texts = _about()
            assert texts[0] == "spaCR"
            assert texts[2] == version_line
            assert any(t.startswith("Python ") for t in texts) is has_build
            assert texts[-1] == "© Olafsson Lab"
            assert any(LICENSE_NAME in t for t in texts)
            assert mark.pixmap().isNull() is (attr == "RESOURCE_DIR")


# ---------------------------------------------------------------------------
# Preferences, theme refresh and the Home rebuild
# ---------------------------------------------------------------------------

def test_preferences_and_theme_changes_rebuild_home_and_survive_failures(
        win, monkeypatch, subtests):
    """Preferences rebuilds Home after the dialog; a theme refresh re-inks the
    dock, re-applies maturity per screen, and every failing step costs itself."""
    from spacr.qt import preferences as prefs

    class _FakeDialog:
        def __init__(self, parent=None, calls=None):
            calls.append(parent)
            self.calls = calls

        def exec(self):
            self.calls.append("exec")
            return 1

    with subtests.test(msg="the dialog opens on the window, Home is rebuilt"), \
            monkeypatch.context() as m:
        calls: list = []
        m.setattr(prefs, "PreferencesDialog",
                  lambda parent=None: _FakeDialog(parent, calls))
        old_home = win._startup
        win._on_nav_selected("__home__")
        win._open_preferences()
        assert calls == [win, "exec"]
        assert win._startup is not old_home
        assert win._stack.indexOf(old_home) == -1
        assert win._stack.currentWidget() is win._startup

    with subtests.test(msg="no dialog: the status bar says so"), \
            monkeypatch.context() as m:
        m.delattr(prefs, "PreferencesDialog")
        old_home = win._startup
        assert win.show_preferences_on() is False
        assert win.statusBar().currentMessage().startswith(
            "Preferences unavailable: cannot import name 'PreferencesDialog'")
        assert win._startup is old_home

    with subtests.test(msg="a Home rebuild failure after the dialog"), \
            monkeypatch.context() as m:
        m.setattr(prefs, "PreferencesDialog",
                  lambda parent=None: _FakeDialog(parent, []))
        m.setattr(win, "_install_startup_page", _boom)
        win._open_preferences()
        assert win._stack.indexOf(win._startup) != -1

    with subtests.test(msg="refresh re-inks the dock and rebuilds Home"), \
            monkeypatch.context() as m:
        inked: list = []
        m.setattr(win._sidebar, "refresh_icons", lambda: inked.append("icons"))
        old_home = win._startup
        win.refresh_theme()
        assert inked == ["icons"]
        assert win._startup is not old_home
        assert win._stack.indexOf(old_home) == -1

    with subtests.test(msg="refresh survives either half failing"), \
            monkeypatch.context() as m:
        m.setattr(win._sidebar, "refresh_icons", _boom)
        old_home = win._startup
        win.refresh_theme()
        assert win._startup is not old_home
        m.setattr(win, "_install_startup_page", _boom)
        still = win._startup
        win.refresh_theme()
        assert win._startup is still

    with subtests.test(msg="refresh asks each screen for its maturity hook"), \
            monkeypatch.context() as m:
        asked = _Recorder()
        hooked = QWidget(win)
        hooked.refresh_maturity_visibility = asked.refresh_maturity_visibility
        plain = QWidget(win)
        m.setattr(win, "_screens", {"with-hook": hooked, "without-hook": plain})
        win.refresh_theme()
        assert asked.names() == ["refresh_maturity_visibility"]
        assert not hasattr(plain, "refresh_maturity_visibility")

    with subtests.test(msg="refresh carries on past every failing step"), \
            monkeypatch.context() as m:
        m.setattr(preferences_mod, "_apply_alpha_widgets", _boom)
        screen = types.SimpleNamespace(refresh_maturity_visibility=_boom,
                                       _refresh_alpha_visibility=_boom)
        languages: list = []
        fake = types.SimpleNamespace(
            findChildren=lambda widget_type: [],
            _sidebar=types.SimpleNamespace(refresh_icons=_boom,
                                           refresh_visibility=_boom),
            _refresh_app_action_visibility=_boom,
            _backdrop_the_dock_column=_boom,
            _screens={"probe": screen},
            _drop_a_redundant_screen_backdrop=_boom,
            apply_dock_mode=_boom,
            _rebuild_startup_page=_boom,
            refresh_language=lambda: languages.append(True))
        assert MainWindow.refresh_theme(fake) is None
        assert languages == [True]


def test_rebuilding_home_leaves_exactly_one_home_page(win, monkeypatch, subtests):
    """The old page leaves the stack and a new one replaces it, without
    changing what is on screen; with no old page one is simply installed."""
    with subtests.test(msg="a non-Home screen stays on screen"):
        win._on_nav_selected("mask")
        current = win._stack.currentWidget()
        old, before = win._startup, win._stack.count()
        win._rebuild_startup_page()
        assert win._startup is not old
        assert win._stack.indexOf(old) == -1
        assert win._stack.count() == before
        assert win._stack.currentWidget() is current

    with subtests.test(msg="an outgoing page that will not close"):
        old, before = win._startup, win._stack.count()
        monkeypatch.setattr(old, "close", _boom)
        win._rebuild_startup_page()
        assert win._startup is not old
        assert win._stack.indexOf(old) == -1
        assert win._stack.count() == before

    with subtests.test(msg="no page to replace"):
        before = win._stack.count()
        win._startup = None
        win._rebuild_startup_page()
        assert win._stack.count() == before + 1
        assert win._stack.indexOf(win._startup) >= 0


# ---------------------------------------------------------------------------
# window state and the menu bar
# ---------------------------------------------------------------------------

def test_a_window_state_change_re_lays_the_menu_bar_or_logs_why_not(
        win, qtbot, app_log, monkeypatch, subtests):
    """A state change re-measures the corner buttons, forgets a drag, and never
    raises: no bar, no corner widget and a broken maximise mark are survived."""
    win.show()
    qtbot.waitExposed(win)
    bar = win.menuBar()
    corner = bar.cornerWidget(Qt.Corner.TopRightCorner)
    state = QWindowStateChangeEvent(Qt.WindowState.WindowNoState)

    with subtests.test(msg="the corner buttons are re-measured"):
        assert corner is not None
        corner.setGeometry(0, 0, 3, 3)
        win._drag_from = (1, 2)
        win.changeEvent(state)
        assert corner.size() == corner.sizeHint()
        assert win._drag_from is None
        assert "the menu bar could not be asked for" not in app_log.text()
        assert "could not schedule the menu-bar re-lay" not in app_log.text()

    with subtests.test(msg="no room folds the dock; room again reopens it"):
        from spacr.qt.preferences import scaled_px
        central, edge = win.centralWidget(), win._dock_edge
        dock = win._sidebar.column_width()
        assert dock > 0
        # Start from an open dock that was not folded for room.
        edge.set_collapsed(False)
        win._dock_fits, win._dock_folded_for_room = None, False
        central.resize(dock * 2, central.height())
        assert (edge.is_collapsed(), win._dock_folded_for_room,
                win._dock_fits) == (True, True, False)
        central.resize(dock * 4 + scaled_px(720), central.height())
        assert (edge.is_collapsed(), win._dock_folded_for_room,
                win._dock_fits) == (False, False, True)
        assert (win._dock_room_change, win._setting_navigator_dock) == (
            False, False)

    with subtests.test(msg="a broken maximise mark"), monkeypatch.context() as m:
        m.setattr(win, "_maximise_button", object())
        win._drag_from = (1, 2)
        win.changeEvent(state)
        assert win._drag_from is None
        assert "could not schedule the menu-bar re-lay" in app_log.text()

    with subtests.test(msg="a bar that cannot be asked for"), \
            monkeypatch.context() as m:
        m.setattr(win, "menuBar", _boom)
        win.changeEvent(state)
        assert "the menu bar could not be asked for" in app_log.text()
        assert len(bar.actions()) > 1
        press = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(4, 4),
                            QPointF(4, 4), Qt.MouseButton.LeftButton,
                            Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier)
        assert win.eventFilter(bar, press) is False
        assert ("the menu bar could not be asked for during an event"
                in app_log.messages)

    with subtests.test(msg="no corner widget, a bar with its own layout"):
        QHBoxLayout(bar)
        bar.setCornerWidget(None, Qt.Corner.TopRightCorner)
        corner.setGeometry(0, 0, 3, 3)
        win.changeEvent(state)
        assert corner.size() == QSize(3, 3)
        assert bar.layout() is not None and len(bar.actions()) > 1


# ---------------------------------------------------------------------------
# Edit menu
# ---------------------------------------------------------------------------

def _screen_with(window, child):
    screen = QWidget()
    QVBoxLayout(screen).addWidget(child)
    window._stack.addWidget(screen)
    window._stack.setCurrentWidget(screen)
    return screen


def test_the_edit_menu_follows_the_focused_document_editor(win, subtests):
    """Undo/Redo act on the focused editor; nothing to undo, a deleted focus
    widget or no editor at all leave both disabled and change nothing."""
    for editor_type in (QTextEdit, QPlainTextEdit):
        with subtests.test(msg=f"{editor_type.__name__} owns the menu"):
            editor = editor_type()
            _screen_with(win, editor)
            editor.setPlainText("one")
            editor.selectAll()
            editor.insertPlainText("two")
            win._edit_focus = editor
            win._refresh_edit_actions()
            assert (win._act_edit_undo.isEnabled(),
                    win._act_edit_redo.isEnabled()) == (True, False)
            win._invoke_edit_action(redo=False)
            assert editor.toPlainText() == "one"
            assert win._act_edit_redo.isEnabled() is True
            win._invoke_edit_action(redo=True)
            assert editor.toPlainText() == "two"

    with subtests.test(msg="an unavailable step is not run"):
        editor = QTextEdit()
        _screen_with(win, editor)
        editor.setPlainText("kept")
        editor.document().clearUndoRedoStacks()
        win._edit_focus = editor
        win._invoke_edit_action(redo=False)
        win._invoke_edit_action(redo=True)
        assert editor.toPlainText() == "kept"
        assert win._act_edit_undo.isEnabled() is False

    with subtests.test(msg="a deleted focus widget is forgotten"):
        _screen_with(win, QTextEdit())
        gone = object()
        win._edit_focus = gone
        page = win._stack.currentWidget()
        original = page.isAncestorOf

        def deleted(widget):
            if widget is gone:
                raise RuntimeError("Internal C++ object already deleted.")
            return original(widget)

        page.isAncestorOf = deleted
        assert win._edit_target() is None
        assert win._edit_focus is None
        win._invoke_edit_action(redo=False)
        assert (win._act_edit_undo.isEnabled(),
                win._act_edit_redo.isEnabled()) == (False, False)


# ---------------------------------------------------------------------------
# closing
# ---------------------------------------------------------------------------

def test_closing_drains_every_screen_and_console_then_ends_the_session_once(
        qtbot, qt_theme_applied, app_log, monkeypatch, subtests):
    """Close shuts every owned screen and console panel down, survives dead
    screens, panels and workers, refuses when a screen fails mid-drain, and
    quits the application exactly once -- never on a refused close."""
    from spacr.qt.widgets.console_panel import ConsolePanel

    with subtests.test(msg="every console panel is drained"), \
            monkeypatch.context() as m:
        window = _window(qtbot)
        window._on_nav_selected("mask")
        window._on_nav_selected("measure")
        panels = window.findChildren(ConsolePanel)
        assert len(panels) >= 2
        drained: list = []
        m.setattr(ConsolePanel, "shutdown", lambda self: drained.append(self))
        window.close()
        assert set(drained) == set(panels) and len(drained) == len(panels)

    survivable = [
        ("a panel that refuses to shut down", "panel"),
        ("an update worker already reaped", "reaped"),
        ("no update check running", "none"),
    ]
    for name, kind in survivable:
        with subtests.test(msg=name), monkeypatch.context() as m:
            window = _window(qtbot)
            window.show()
            if kind == "panel":
                window._on_nav_selected("mask")
                m.setattr(ConsolePanel, "shutdown", _boom)
            elif kind == "reaped":
                window._update_worker = types.SimpleNamespace(
                    wait=lambda msecs=None: _boom())
            else:
                assert not hasattr(window, "_update_worker")
            assert window.close() is True
            assert window.isVisible() is False
            assert window._closing is True

    with subtests.test(msg="a screen deleted under us"), \
            monkeypatch.context() as m:
        from shiboken6 import delete, isValid
        window = _window(qtbot)
        gone = QWidget(window)
        live = _StubScreen("live", parent=window)
        delete(gone)
        assert not isValid(gone)
        m.setattr(window, "_screens", {"a": gone, "b": live})
        m.setattr(app_mod, "QApplication",
                  types.SimpleNamespace(instance=lambda: None))
        event = QCloseEvent()
        window.closeEvent(event)
        assert live.closed == ["live"]
        assert (event.isAccepted(), window._closing) == (True, True)

    with subtests.test(msg="a screen that raises keeps the window open"), \
            monkeypatch.context() as m:
        window = _window(qtbot)
        angry = _StubScreen("angry", ValueError("mid-write"), parent=window)
        later = _StubScreen("later", parent=window)
        m.setattr(window, "_screens", {"a": angry, "b": later})
        event = QCloseEvent()
        window.closeEvent(event)
        assert (event.isAccepted(), window._closing) == (False, False)
        assert (angry.closed, later.closed) == (["angry"], [])
        assert "Could not close an owned application screen" in app_log.text()

    with subtests.test(msg="a screen that vetoes keeps the window open"), \
            monkeypatch.context() as m:
        window = _window(qtbot)
        veto = _StubScreen("veto", False, parent=window)
        m.setattr(window, "_screens", {"a": veto})
        event = QCloseEvent()
        window.closeEvent(event)
        assert (event.isAccepted(), window._closing) == (False, False)
        assert "Shutdown deferred by an application screen" in app_log.text()

    with subtests.test(msg="only an accepted close quits, exactly once"), \
            monkeypatch.context() as m:
        from PySide6.QtCore import QTimer
        window = _window(qtbot)
        quits = _Recorder()
        holder = {"instance": quits}
        m.setattr(app_mod, "QApplication", types.SimpleNamespace(
            instance=lambda: holder["instance"]))
        m.setattr(app_mod, "_an_event_loop_is_running", lambda: True)
        queued: list = []
        m.setattr(QTimer, "singleShot",
                  lambda delay, callback: queued.append((delay, callback)))
        window.closeEvent(_RefusedClose())
        assert quits.names() == []
        first = QCloseEvent()
        window.closeEvent(first)
        assert first.isAccepted() is True
        if sys.platform == "darwin":
            assert [d for d, _cb in queued] == [0]
            queued.pop()[1]()
            ended = ["closeAllWindows", "exit"]
        else:
            assert queued == []
            ended = ["quit"]
        assert quits.names() == ended
        holder["instance"] = None
        window.closeEvent(QCloseEvent())
        assert quits.names() == ended, "no application, nothing to quit"

    with subtests.test(msg="on macOS the loop is left without asking Cocoa"), \
            monkeypatch.context() as m:
        window = _window(qtbot)
        calls: list = []
        m.setattr(app_mod.sys, "platform", "darwin")
        m.setattr(app_mod, "_an_event_loop_is_running", lambda: True)
        application = types.SimpleNamespace(
            quit=lambda: calls.append("quit"),
            closeAllWindows=lambda: calls.append("close all"),
            exit=lambda code: calls.append(("exit", code)))
        m.setattr(app_mod, "QApplication",
                  types.SimpleNamespace(instance=lambda: application))
        window.closeEvent(_RefusedClose())
        assert calls == []
        window.closeEvent(QCloseEvent())
        window.closeEvent(QCloseEvent())
        assert calls == ["close all"]
        qtbot.waitUntil(lambda: ("exit", 0) in calls)
        from PySide6.QtWidgets import QApplication as QtApplication
        QtApplication.processEvents()
        assert calls == ["close all", ("exit", 0)], "the loop is left once"
