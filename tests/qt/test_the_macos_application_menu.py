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
#: The claim is about ORDER -- that the names are set before the application
#: object exists -- and a process that already has a ``QApplication`` cannot
#: observe that moment. It also keeps the application-wide rename out of the
#: session's own Qt state.
_NAME_PROBE = textwrap.dedent('''
    import json, os, sys
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["SPACR_NO_SETUP"] = "1"

    from PySide6.QtCore import QCoreApplication
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    import spacr.qt.app as app_mod

    seen = {"module": app_mod.__file__}

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
    print("RESULT " + json.dumps(seen))
''')


def _repo_root() -> str:
    import spacr

    return os.path.dirname(os.path.dirname(os.path.abspath(spacr.__file__)))


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
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _NAME_PROBE],
        cwd=root, env=env, capture_output=True, text=True, timeout=240)
    line = next((ln for ln in proc.stdout.splitlines()
                 if ln.startswith("RESULT ")), None)
    assert line, f"launch() produced no result\n{proc.stdout}\n{proc.stderr}"
    seen = json.loads(line[len("RESULT "):])
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

def test_the_bundle_name_patch_is_macos_only():
    """It must be inert, not merely harmless, on this platform."""
    from spacr.qt.menus import name_the_macos_application_menu

    assert sys.platform != "darwin", "this assertion is about the other case"
    assert name_the_macos_application_menu("spaCR") is False


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
    action.trigger()
    qtbot.waitUntil(lambda: not window.isFullScreen(), timeout=3000)
