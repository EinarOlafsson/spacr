"""The macOS application menu: its name, and a route into it that stays put.

macOS MOVES an action carrying ``PreferencesRole`` or ``QuitRole`` out of
whatever menu it was added to and into the application menu -- the one beside
the Apple logo. That is the platform convention and spaCR keeps it, which is
why Preferences and Quit are absent from spaCR's own dropdown there. A user
who looks in that dropdown and finds neither is watching the convention work,
and is looking in the only place they had a reason to look.

TWO THINGS DECIDE WHERE "SOMEWHERE" IS.

The first is the application's NAME. macOS titles that menu from the running
bundle's ``CFBundleName``, and Qt builds its contents ("About X", "Quit X")
from ``QCoreApplication::applicationName`` -- both read while the Cocoa
platform plugin comes up INSIDE the ``QApplication`` constructor. Naming the
application on the next line is too late for the menu and looks exactly like
success from every other angle: the Qt-side menu bar, built later, sees the
right name either way. So the assertion here is taken at construction, which
is the moment that matters, and it is taken by driving the real ``launch()``.

The second is a route macOS does not touch. Even correctly named, the
application menu is not where a user goes looking, so Help carries a Window
submenu holding copies with ``NoRole`` pinned -- separate ``QAction`` objects
from the ones macOS relocates, because a role belongs to an action and one
action cannot be in two places at once. The same submenu carries minimise,
maximise and close, which the frameless window otherwise offers only as marks
in the menu bar's corner, and a corner widget that the platform lays out
where nothing shows it leaves the window with no visible control at all.

None of the relocation is observable on Linux, where Qt applies menu roles
not at all. What IS observable, and is what was actually wrong, is asserted
here: the names at construction time, and that nothing in the second route
carries a role that would let macOS move it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QAction, QKeySequence      # noqa: E402
from PySide6.QtWidgets import QMenu                  # noqa: E402

pytestmark = pytest.mark.qt


# --------------------------------------------------------------------------
# The name, measured where it counts: at QApplication construction.
# --------------------------------------------------------------------------

#: Driven through `launch()` in a child process rather than in-process.
#:
#: The child prints its result flushed and leaves with ``os._exit``. Its
#: stubbed ``exec()`` returns without quitting, so ``aboutToQuit`` never
#: drains the jobs the window started, and interpreter shutdown then
#: destroyed a still-running bridge QThread: an abort or segfault after the
#: result had been written to a pipe buffer that was never flushed, which
#: the N47 serial Qt run reported as "launch() produced no result".
#:
#: The claim is about ORDER -- that the names are set before the application
#: object exists -- and a process that already has a ``QApplication`` cannot
#: observe that moment. It also keeps the application-wide rename out of the
#: session's own Qt state.
_OWNED_SETTINGS_PROBE = textwrap.dedent('''
    import json, os, sys
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["SPACR_NO_SETUP"] = "1"

    from PySide6 import QtCore
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    NativeSettings = QtCore.QSettings
    settings_root = os.environ["SPACR_MENU_TEST_SETTINGS"]
    for scope in (NativeSettings.UserScope, NativeSettings.SystemScope):
        NativeSettings.setPath(NativeSettings.IniFormat, scope, settings_root)

    class OwnedSettings(NativeSettings):
        def __init__(self, *args, **kwargs):
            if args and isinstance(args[0], str) and not (
                    len(args) > 1 and isinstance(args[1], NativeSettings.Format)):
                args = (NativeSettings.IniFormat, NativeSettings.UserScope) + args
            elif args and isinstance(args[0], NativeSettings.Scope):
                args = (NativeSettings.IniFormat,) + args
            elif args and args[0] == NativeSettings.NativeFormat and len(args) > 2:
                args = (NativeSettings.IniFormat,) + args[1:]
            elif (not args or isinstance(args[0], QtCore.QObject)) and not kwargs:
                args = (NativeSettings.IniFormat, NativeSettings.UserScope,
                        QCoreApplication.organizationName(), QCoreApplication.applicationName()) + args
            super().__init__(*args, **kwargs)
            assert os.path.commonpath([os.path.realpath(self.fileName()), settings_root]) == settings_root

    QtCore.QSettings = OwnedSettings
    settings_file = OwnedSettings("spacr", "qt").fileName()
    assert os.path.commonpath([os.path.realpath(settings_file), settings_root]) == settings_root
''')

_NAME_PROBE = _OWNED_SETTINGS_PROBE + textwrap.dedent('''
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    import spacr.qt.app as app_mod

    seen = {"module": app_mod.__file__,
            "settings": settings_file}

    def names():
        return [QCoreApplication.applicationName(),
                QGuiApplication.applicationDisplayName()]

    class RecordingApplication(QApplication):
        """Records the names the constructor was handed, before running it."""

        def __init__(self, argv):
            seen["at_construction"] = names()
            super().__init__(argv)

        def exec(self):
            seen["at_exec"] = names()
            return 0

    class RecordingWindow(app_mod.MainWindow):
        def _build_menu_bar(self):
            seen["at_menu_bar"] = names()
            return super()._build_menu_bar()

    app_mod.QApplication = RecordingApplication
    app_mod.MainWindow = RecordingWindow
    seen["rc"] = app_mod.launch(["--no-setup"])
    print("RESULT " + json.dumps(seen), flush=True)
    os._exit(0)
''')


def _repo_root() -> str:
    import spacr

    return os.path.dirname(os.path.dirname(os.path.abspath(spacr.__file__)))


_SCALE_STARTUP_PROBE = _OWNED_SETTINGS_PROBE + textwrap.dedent('''
    import runpy, traceback
    import numpy as np
    from PySide6.QtCore import QPoint, QRect, QSize, QTimer
    from PySide6.QtGui import QAction, QImage
    from PySide6.QtWidgets import QAbstractButton, QLabel, QStyle
    from spacr.qt import app as app_mod, gui_scale, preferences, theme, timing
    from spacr.logging_util import _spacr_home

    assert os.path.commonpath([str(_spacr_home().resolve()), settings_root]) == settings_root

    result = {"module": app_mod.__file__, "settings": settings_file, "metrics": []}

    def measure(window, gui=.9, font=.9):
        result["preferences_settings"] = preferences._settings().fileName()
        assert os.path.commonpath([os.path.realpath(result["preferences_settings"]), settings_root]) == settings_root
        assert window.isVisible() and window._startup.isVisible()
        assert window._stack.currentWidget() is window._startup
        buttons = [button for button in window.findChildren(QAbstractButton)
                   if button.isVisible() and button.objectName() == "AppTile"
                   and button.visibleRegion().boundingRect() == button.rect()]
        assert buttons, [(type(button).__name__, button.objectName(), button.isVisible(),
                          button.icon().isNull()) for button in window.findChildren(QAbstractButton)]
        button = sorted(buttons, key=lambda item: item.property("moduleAppKey"))[0]
        glyphs = [label for label in button.findChildren(QLabel) if not label.pixmap().isNull()]
        assert len(glyphs) == 1
        glyph = glyphs[0]
        pixmap = glyph.pixmap()
        size = pixmap.deviceIndependentSize()
        assert size.width() > 0 and size.height() > 0
        assert glyph.visibleRegion().boundingRect() == glyph.rect()
        assert button.name_label.visibleRegion().boundingRect() == button.name_label.rect()
        assert button.rect().contains(QRect(glyph.mapTo(button, QPoint()), glyph.size()))
        image = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
        pixels = np.frombuffer(image.bits(), np.uint8).reshape(image.height(), image.bytesPerLine())
        ys, xs = np.nonzero(pixels[:, :image.width()*4].reshape(image.height(), image.width(), 4)[..., 3])
        assert len(xs), "Production icon has no visible pixels"
        painted = QStyle.alignedRect(glyph.layoutDirection(), glyph.alignment(),
                                    QSize(round(size.width()), round(size.height())), glyph.contentsRect())
        dpr = pixmap.devicePixelRatio()
        ink = [painted.x()+xs.min()/dpr, painted.y()+ys.min()/dpr,
               painted.x()+(xs.max()+1)/dpr, painted.y()+(ys.max()+1)/dpr]
        contents = glyph.contentsRect()
        overflow = (ink[0] < contents.x() or ink[1] < contents.y()
                    or ink[2] > contents.x()+contents.width() or ink[3] > contents.y()+contents.height())
        details = dict(key=button.property("moduleAppKey"), pixmap=[size.width(), size.height()],
                       contents=glyph.contentsRect().getRect(), dpr=pixmap.devicePixelRatio(),
                       alpha=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                       alignment=int(glyph.alignment()), visible=glyph.visibleRegion().boundingRect().getRect(),
                       painted=painted.getRect(), ink=[float(value) for value in ink], ink_overflows=bool(overflow))
        assert size.width() <= glyph.contentsRect().width(), details
        assert size.height() <= glyph.contentsRect().height(), details
        assert gui_scale.current_scale() == gui
        assert preferences.get_gui_scale() == gui and preferences.get_font_scale() == font
        assert button.name_label.font().pixelSize() == gui_scale.scale_int(theme.font_px(14, font), gui)
        return dict(key=button.property("moduleAppKey"), font=button.name_label.font().pixelSize(),
                    icon=[size.width(), size.height()],
                    dpr=pixmap.devicePixelRatio(),
                    pixels=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])

    class MeasuredWindow(app_mod.MainWindow):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            timing.subscribe_readiness(self.ready)
            self.started = False
            QTimer.singleShot(45000, self.deadline)

        def ready(self, entry):
            if (entry.get("detail") == "__home__" and not self.started
                    and self.isVisible() and self._startup.isVisible()):
                self.started = True
                timing.unsubscribe_readiness(self.ready)
                QTimer.singleShot(0, self.check_startup)

        def deadline(self):
            if not self.started:
                result["error"] = "Normal entrypoint did not reach post-paint interactive Home within45s"
                self.finish()

        def check_startup(self):
            try:
                result["metrics"].append(measure(self))
                if os.environ["SPACR_SCALE_TEST_PHASE"] == "reset":
                    gui_scale.change_scales(self, gui=1.25, font=1.5, ask=False)
                    assert gui_scale.current_scale() == 1.25
                    assert preferences.get_font_scale() == 1.5
                    result["changed"] = measure(self, 1.25, 1.5)
                    assert result["changed"]["font"] != result["metrics"][0]["font"]
                    assert result["changed"]["icon"] != result["metrics"][0]["icon"]
                    gui_scale.reset_every_scale(self)
                    QTimer.singleShot(500, self.check_reset)
                else:
                    self.finish()
            except Exception:
                result["error"] = traceback.format_exc()
                self.finish()

        def check_reset(self):
            try:
                result["metrics"].append(measure(self))
                assert result["metrics"][0] == result["metrics"][1]
                preferences._settings().sync()
            except Exception:
                result["error"] = traceback.format_exc()
            self.finish()

        def finish(self):
            print("RESULT " + json.dumps(result), flush=True)
            actions = [action for action in self._menu_bar_actions()
                       if action.menuRole() == QAction.QuitRole]
            assert len(actions) == 1
            QTimer.singleShot(0, actions[0].trigger)

    app_mod.MainWindow = MeasuredWindow
    sys.argv = ["spacr.qt", "__home__", "--no-setup"]
    runpy.run_module("spacr.qt", run_name="__main__")
''')


@pytest.mark.skipif(sys.platform != "darwin", reason="Actual Cocoa startup/reset/restart contract")
def test_normal_macos_startup_reset_and_restart_keep_scale_font_and_icons(tmp_path):
    root = _repo_root()
    settings = str((tmp_path / "settings").resolve())
    state = os.path.join(settings, "state")
    cache = os.path.join(settings, "cache")
    env = dict(os.environ, QT_QPA_PLATFORM="cocoa", CUDA_VISIBLE_DEVICES="",
               SPACR_MENU_TEST_SETTINGS=settings, SPACR_NO_BACKDROP="1",
               SPACR_NO_NETWORK="1", SPACR_LANGUAGE="en", PYTHONPATH=root,
               SPACR_HOME=state, SPACR_LOG_DIR=os.path.join(state, "logs"),
               SPACR_ISOLATED_PROFILE="1", SPACR_LOCK_DIR=os.path.join(state, "locks"),
               SPACR_CHAINING_PINS=os.path.join(state, "pins.json"),
               SPACR_BACKENDS_DIR=os.path.join(state, "backends"),
               SPACR_MODEL_CACHE=os.path.join(state, "models"),
               SPACR_PLUGIN_HOME=os.path.join(state, "plugins"),
               SPACR_SOUND_CACHE=os.path.join(cache, "sounds"),
               XDG_CACHE_HOME=cache, XDG_STATE_HOME=os.path.join(state, "xdg"),
               TORCH_HOME=os.path.join(cache, "torch"), HF_HOME=os.path.join(cache, "huggingface"),
               MPLCONFIGDIR=os.path.join(cache, "matplotlib"),
               CELLPOSE_LOCAL_MODELS_PATH=os.path.join(state, "models", "cellpose"),
               SPACR_TIMING="1", SPACR_TIMING_IMPORTS="0")
    reports = []
    for phase in ("reset", "restart"):
        result = subprocess.run([sys.executable, "-c", _SCALE_STARTUP_PROBE], cwd=root,
                                env=dict(env, SPACR_SCALE_TEST_PHASE=phase),
                                capture_output=True, text=True, timeout=130)
        assert result.returncode == 0, result.stdout + result.stderr
        lines = [line for line in result.stdout.splitlines() if line.startswith("RESULT ")]
        assert len(lines) == 1, result.stdout + result.stderr
        report = json.loads(lines[0][len("RESULT "):])
        assert "error" not in report, report.get("error")
        assert report["module"] == os.path.join(root, "spacr", "qt", "app.py")
        assert os.path.commonpath([os.path.realpath(report["settings"]), settings]) == settings
        assert os.path.commonpath([os.path.realpath(report["preferences_settings"]), settings]) == settings
        reports.append(report)
    assert len(reports[0]["metrics"]) == 2 and len(reports[1]["metrics"]) == 1
    assert reports[0]["metrics"][0] == reports[0]["metrics"][1] == reports[1]["metrics"][0]


@pytest.fixture(scope="module")
def launched(tmp_path_factory):
    """Run the real ``launch()`` once, with ``exec()`` stubbed out."""
    root = _repo_root()
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["SPACR_NO_SETUP"] = "1"
    # Qt writes preferences through QSettings; keep them out of the real one.
    env["XDG_CONFIG_HOME"] = str(tmp_path_factory.mktemp("config"))
    env["SPACR_MENU_TEST_SETTINGS"] = str(tmp_path_factory.mktemp("menu-settings").resolve())
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _NAME_PROBE],
        cwd=root, env=env, capture_output=True, text=True, timeout=240)
    line = next((ln for ln in proc.stdout.splitlines()
                 if ln.startswith("RESULT ")), None)
    assert line, f"launch() produced no result\n{proc.stdout}\n{proc.stderr}"
    seen = json.loads(line[len("RESULT "):])
    assert os.path.commonpath([os.path.realpath(seen["settings"]),
                               env["SPACR_MENU_TEST_SETTINGS"]]) == env["SPACR_MENU_TEST_SETTINGS"]
    # The tree under test, not an installed copy of it.
    assert seen["module"] == os.path.join(root, "spacr", "qt", "app.py")
    return seen


def test_both_names_are_spacr_before_the_application_exists(launched):
    """The one that was broken.

    Before the fix this read ``["", ""]``: nothing had named the application
    when its constructor ran, so Qt fell back to ``argv[0]`` -- the console
    script, the script file, or ``PySideApp`` for an empty argument list --
    and that is the name macOS built its application menu from.
    """
    assert launched["at_construction"] == ["spaCR", "spaCR"], (
        "the QApplication was constructed before the application had a name; "
        "on macOS the application menu is built inside that constructor")


def test_both_names_are_spacr_when_the_menu_bar_is_built(launched):
    """Weaker than the test above, and it is the one that always passed.

    Named after construction, the Qt menu bar still sees "spaCR" because it
    is built later still. Kept as the control: if this were the only
    assertion, the bug would have been reported fixed.
    """
    assert launched["at_menu_bar"] == ["spaCR", "spaCR"]


def test_the_display_name_is_set_at_all(launched):
    """``applicationDisplayName`` was never assigned anywhere.

    It has no default of its own beyond mirroring ``applicationName``, and it
    is the one Qt shows to people rather than the one it keys settings on.
    """
    assert launched["at_exec"][1] == "spaCR"


def test_the_launch_survives_being_named_early(launched):
    """Naming before construction must not cost the launch itself."""
    assert launched["rc"] == 0


# --------------------------------------------------------------------------
# The bundle name macOS titles the menu from.
# --------------------------------------------------------------------------

def test_the_bundle_name_patch_is_macos_only(monkeypatch):
    """It must be inert, not merely harmless, on this platform."""
    from spacr.qt import menus

    monkeypatch.setattr(menus.sys, "platform", "linux")
    assert menus.name_the_macos_application_menu("spaCR") is False


def test_the_bundle_name_patch_never_raises(monkeypatch):
    """A cosmetic menu title is not worth a failed start.

    Driven with the platform claiming to be macOS and the Objective-C runtime
    unreachable, which is the shape of every way this can go wrong on a
    future macOS: report failure, do not propagate it.
    """
    import ctypes.util

    from spacr.qt import menus

    monkeypatch.setattr(menus.sys, "platform", "darwin")
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: None)
    assert menus.name_the_macos_application_menu("spaCR") is False


def test_the_bundle_name_patch_can_be_switched_off(monkeypatch):
    """Without a code change, because a Mac is where it would misbehave."""
    from spacr.qt import menus

    monkeypatch.setattr(menus.sys, "platform", "darwin")
    monkeypatch.setenv(menus.BUNDLE_NAME_OPT_OUT, "1")
    assert menus.name_the_macos_application_menu("spaCR") is False


def test_the_opt_out_is_off_by_default(monkeypatch):
    """An unset or ``0`` value must not read as "switched off"."""
    import ctypes.util

    from spacr.qt import menus

    calls = []
    monkeypatch.setattr(menus.sys, "platform", "darwin")
    monkeypatch.setenv(menus.BUNDLE_NAME_OPT_OUT, "0")
    monkeypatch.setattr(ctypes.util, "find_library",
                        lambda name: calls.append(name) or None)
    assert menus.name_the_macos_application_menu("spaCR") is False
    assert calls, "the opt-out was read as set when it holds '0'"


# --------------------------------------------------------------------------
# The route macOS does not relocate.
# --------------------------------------------------------------------------

@pytest.fixture
def window(qtbot, qt_theme_applied):
    """A main window with every late menu contributor installed.

    ``recipes``, ``walkthrough`` and ``feature_dictionary`` all add to Help
    after ``_build_menu_bar`` has run, and the central role sweep has to hold
    with them present.
    """
    from spacr.qt.app import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    from spacr.qt import shortcuts
    shortcuts.install(win)
    return win


def _window_menu(win) -> QMenu:
    menus = [m for m in win.menuBar().findChildren(QMenu)
             if m.title().replace("&", "") == "Window"]
    assert len(menus) == 1, [m.title() for m in win.menuBar().findChildren(QMenu)]
    return menus[0]


def _entries(menu) -> dict:
    return {a.text(): a for a in menu.actions() if not a.isSeparator()}


def test_the_window_menu_hangs_off_help(window):
    """Reachable by the route the report named, not merely present."""
    help_menu = next(m for m in window.menuBar().findChildren(QMenu)
                     if m.title().replace("&", "") == "Help")
    submenus = [a.menu().title() for a in help_menu.actions()
                if a.menu() is not None]
    assert "Window" in submenus, submenus


def test_the_second_route_reaches_preferences_and_quit(window):
    entries = _entries(_window_menu(window))
    assert "Preferences…" in entries
    assert "Quit" in entries


def test_the_copy_is_noroled_and_the_original_keeps_its_role(window):
    """The point of the copy. A role belongs to an action, and one action
    cannot be in two menus at once -- so the copy has to be a different
    object, or pinning it ``NoRole`` would unpin the original."""
    entries = _entries(_window_menu(window))
    for text in ("Preferences…", "Quit"):
        assert entries[text].menuRole() == QAction.MenuRole.NoRole, (
            f"macOS would relocate the {text!r} copy as well")
    assert entries["Preferences…"] is not window._act_preferences
    assert entries["Quit"] is not window._act_quit
    assert (window._act_preferences.menuRole()
            == QAction.MenuRole.PreferencesRole)
    assert window._act_quit.menuRole() == QAction.MenuRole.QuitRole


def test_nothing_in_the_window_menu_can_be_relocated(window):
    """Including the menu's own action, which opens it."""
    menu = _window_menu(window)
    for action in list(menu.actions()) + [menu.menuAction()]:
        if action.isSeparator() or not action.text():
            continue
        assert action.menuRole() == QAction.MenuRole.NoRole, action.text()


def test_the_roles_stay_as_measured(window):
    """Exactly one Preferences, one Quit, one About; everything else NoRole.

    Asserted after every module that adds to the menus has done so, and with
    the copies in place -- adding a second action reading "Preferences…" is
    exactly how a sweep keyed on text rather than identity would break.
    """
    special = [(a.text(), a.menuRole()) for a in window._menu_bar_actions()
               if a.menuRole() != QAction.MenuRole.NoRole]
    assert sorted(special) == sorted([
        ("Preferences…", QAction.MenuRole.PreferencesRole),
        ("Quit", QAction.MenuRole.QuitRole),
        ("About spaCR", QAction.MenuRole.AboutRole),
    ]), special


# --------------------------------------------------------------------------
# The window controls, for a frame that is not always drawn.
# --------------------------------------------------------------------------

def test_minimise_maximise_and_close_are_all_on_the_menu(window):
    entries = _entries(_window_menu(window))
    for text in ("Minimise", "Maximise", "Full screen", "Close window"):
        assert text in entries, sorted(entries)


def test_maximise_toggles_the_real_window_state(qtbot, window):
    """Driven through the menu action, and the window state measured."""
    window.show()
    qtbot.waitExposed(window)
    entries = _entries(_window_menu(window))
    assert window.isMaximized() is False
    entries["Maximise"].trigger()
    qtbot.waitUntil(lambda: window.isMaximized(), timeout=3000)
    entries["Maximise"].trigger()
    qtbot.waitUntil(lambda: not window.isMaximized(), timeout=3000)


def test_minimise_really_minimises(qtbot, window):
    window.show()
    qtbot.waitExposed(window)
    _entries(_window_menu(window))["Minimise"].trigger()
    qtbot.waitUntil(lambda: window.isMinimized(), timeout=3000)
    window.showNormal()


def test_close_window_closes_it(qtbot, window):
    window.show()
    qtbot.waitExposed(window)
    _entries(_window_menu(window))["Close window"].trigger()
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=3000)


def test_exactly_one_action_binds_f11(window):
    """The menu copy is the SAME action, not a second one.

    Two distinct QActions carrying F11 on one window is an ambiguous
    shortcut overload, and Qt answers one by firing neither -- so a menu
    entry advertising the key would have taken the key away.
    """
    bound = {id(a) for a in window.findChildren(QAction)
             if a.shortcut() == QKeySequence("F11")}
    assert len(bound) == 1, [a.text() for a in window.findChildren(QAction)
                             if a.shortcut() == QKeySequence("F11")]


def test_f11_still_toggles_fullscreen(qtbot, window):
    window.show()
    qtbot.waitExposed(window)
    action = next(a for a in window.findChildren(QAction)
                  if a.shortcut() == QKeySequence("F11"))
    action.trigger()
    qtbot.waitUntil(lambda: window.isFullScreen(), timeout=3000)
    assert window.isFullScreen()
    action.trigger()
    qtbot.waitUntil(lambda: not window.isFullScreen(), timeout=3000)
    assert not window.isFullScreen()
