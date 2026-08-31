"""macOS drew no window buttons and split the spaCR menu in two.

Reported from a Mac: "the custom toolbar with minimize maximise and close
are gone", and "the spacr drop down menu is split between a python and
spacr drop down".

Both are ONE default. Qt sets ``nativeMenuBar`` on darwin, which lifts the
whole bar into the system strip, and a native menu bar:

* draws no corner widget -- and the minimise, full screen and close marks
  are this bar's top-right corner widget, so on macOS they did not exist.
  The window is frameless on every platform, so that left a Mac with no
  window buttons at all;
* hoists Preferences, About and Quit into the application menu, titled
  "Python" for an unbundled launch, leaving a second half-empty "spaCR"
  menu beside it;
* cannot be dragged, and this bar is the frameless window's title bar.

Turning it off gives macOS the same bar as Linux. The comment that used to
sit in ``_build_menu_bar`` said the relocation "cannot be overridden";
these tests are the standing evidence that it can.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMenu

from spacr.qt import app as app_mod


@pytest.fixture
def window(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    made = app_mod.MainWindow()
    qtbot.addWidget(made)
    return made


def _spacr_menu(window) -> QMenu:
    """The top-level spaCR menu, found by label with the mnemonic dropped."""
    for action in window.menuBar().actions():
        if action.text().replace("&", "") == "spaCR":
            return action.menu()
    raise AssertionError("no spaCR menu on the bar")


@pytest.fixture
def native_calls(monkeypatch):
    """Every ``setNativeMenuBar`` call made while a window is built.

    ASSERTING THE CALL, NOT THE RESULTING FLAG. The obvious test --
    build under a faked darwin and check ``isNativeMenuBar() is False``
    -- is VACUOUS on Linux, where the flag is already False and stays
    False with the fix deleted. It was written that way first and passed
    against a deliberately broken source, which is the only reason this
    fixture exists. Qt decides the default from the REAL platform at
    widget construction, so nothing a test can fake will move it; what
    can be observed is whether spaCR asked.
    """
    from PySide6.QtWidgets import QMenuBar

    seen: list[bool] = []
    original = QMenuBar.setNativeMenuBar

    def recording(self, value):
        seen.append(bool(value))
        return original(self, value)

    monkeypatch.setattr(QMenuBar, "setNativeMenuBar", recording)
    return seen


def test_a_mac_is_told_to_use_the_in_window_menu_bar(
        qtbot, qt_theme_applied, tmp_path, monkeypatch, native_calls):
    """On darwin, spaCR turns the native menu bar off."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(app_mod.sys, "platform", "darwin")
    qtbot.addWidget(app_mod.MainWindow())
    assert False in native_calls, (
        "the menu bar was never taken off the system strip, so a Mac keeps "
        "the split menu and loses the window marks")


def test_no_other_platform_is_told_anything_about_it(
        qtbot, qt_theme_applied, tmp_path, monkeypatch, native_calls):
    """The other half of the gate: Linux and Windows are left alone.

    Without this the fixture above would still pass if the call were made
    unconditionally, and an unconditional call is a change to two
    platforms that did not ask for one.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(app_mod.sys, "platform", "linux")
    qtbot.addWidget(app_mod.MainWindow())
    assert native_calls == []


def test_the_window_marks_live_in_the_bar_a_mac_now_draws(window):
    """WHY the above matters: a native menu bar draws no corner widget.

    A corner widget can be SET on one without error -- it simply never
    appears. So this asserts what the corner holds, and the tests above
    assert that the bar holding it is the in-window one.
    """
    from PySide6.QtCore import Qt

    corner = window.menuBar().cornerWidget(Qt.Corner.TopRightCorner)
    assert corner is not None
    names = {child.objectName() for child in corner.children()}
    assert {"MinimiseWindow", "FullScreenToggle", "CloseWindow"} <= names


def test_minimise_and_maximise_sit_in_the_spacr_menu_above_quit(window):
    """Asked for directly: "add maximize minimize to the spacr menue above
    quit".

    Order is asserted, not just membership -- "above Quit" is the request,
    and a menu that lists them below it satisfies membership.
    """
    labels = [a.text() for a in _spacr_menu(window).actions()]
    assert "Minimise" in labels and "Maximise" in labels and "Quit" in labels
    assert labels.index("Minimise") < labels.index("Quit")
    assert labels.index("Maximise") < labels.index("Quit")


def test_the_spacr_menu_shows_the_very_actions_the_window_menu_does(window):
    """ONE QAction each, in two menus -- not two objects that agree today.

    This is the property that keeps them from drifting: Qt shares the
    label, the enabled state and the tick across every menu an action
    appears in. Identity is asserted with ``is``, because two QActions
    with equal text would pass any label comparison and still be the bug.

    It also avoids the trap this file's subject already documents for
    F11: a duplicated shortcut is an ambiguous overload, and Qt resolves
    it by firing neither.
    """
    in_spacr = {a.text(): a for a in _spacr_menu(window).actions()}
    in_window = {a.text(): a for a in window._window_menu.actions()}
    for label in ("Minimise", "Maximise"):
        assert in_spacr[label] is in_window[label]


def test_minimise_from_the_spacr_menu_actually_minimises(window, qtbot):
    """The action is driven, not just found.

    A menu entry wired to nothing looks identical to a working one in
    every structural assertion above.
    """
    window.show()
    qtbot.waitExposed(window)
    action = {a.text(): a for a in _spacr_menu(window).actions()}["Minimise"]
    action.trigger()
    assert window.isMinimized()
