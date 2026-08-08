"""A gear beside Copy console, on every module screen.

Asked for on Mask, Measure and Classify. Built into `AppScreen`, so every
module gets it — the reason it is wanted is not specific to those three:
a user notices the font is too small, or the backdrop is costing frames,
while looking at a module, and the alternative is a trip out to the menu
bar.
"""

from __future__ import annotations

import pytest

from PySide6.QtWidgets import QPushButton


@pytest.mark.parametrize("app_key", ["mask", "measure", "classify",
                                     "ml_analyze", "regression"])
def test_every_module_screen_has_a_preferences_gear(qtbot, app_key):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key)
    qtbot.addWidget(screen)

    button = getattr(screen, "_btn_preferences", None)
    assert button is not None, f"{app_key} has no preferences button"
    assert not button.icon().isNull(), "the gear has no icon"
    assert button.accessibleName() == "Preferences", (
        "an icon-only button with no accessible name is unreachable by a "
        "screen reader")


def test_it_sits_to_the_right_of_copy_console(qtbot):
    """Asked for by position, so the position is what is checked."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    screen.resize(1400, 900)
    screen.show()
    qtbot.waitExposed(screen)

    copy = screen._btn_copy_console
    gear = screen._btn_preferences
    assert copy.parentWidget() is gear.parentWidget(), "not on the same row"
    assert gear.x() > copy.x(), (
        f"the gear is at x={gear.x()}, left of Copy console at x={copy.x()}")


def test_it_opens_the_window_dialog_when_there_is_one(qtbot, monkeypatch):
    """The same dialog the menu opens, not a second one.

    A separate instance would apply preferences through a different path,
    and "changed it in the module but the menu still shows the old value"
    is exactly the kind of bug that is never reported clearly.
    """
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)

    opened = []

    class _Window:
        def _open_preferences(self):
            opened.append(True)

    monkeypatch.setattr(screen, "window", lambda: _Window())
    screen._open_preferences_dialog()
    assert opened == [True]


def test_a_screen_with_no_main_window_does_not_raise(qtbot, monkeypatch):
    """Which is how every test builds one, and how a detached screen lives."""
    from spacr.qt.screens import app_screen as module

    screen = module.AppScreen("mask")
    qtbot.addWidget(screen)

    shown = []

    class _Dialog:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            shown.append(True)

    monkeypatch.setattr(screen, "window", lambda: object())
    monkeypatch.setattr("spacr.qt.preferences.PreferencesDialog", _Dialog)
    screen._open_preferences_dialog()
    assert shown == [True]
