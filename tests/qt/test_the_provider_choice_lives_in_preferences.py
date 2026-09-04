"""Which assistant to use is a preference, not a control on every screen.

WHAT THIS REPLACES. `test_regression_provider_menu_surface.py` existed to
prove that a "▾" chevron beside the AI switch did not paint a black plate:
it was a QToolButton, and without a rule of its own Qt's native dark-theme
primitive filled most of its rectangle with the palette's pure-black Window
colour. That was a real bug and the test was a good one -- but the control it
guarded is gone, so the file is replaced rather than left asserting the
appearance of something nobody builds.

WHY THE CONTROL WENT. It put a PREFERENCE -- which assistant do I use -- in
the place where per-run choices are made, and repeated it on the actions row
of every module. It has one answer for the whole application. Worse, the
answer did not persist: `preferences.get_preferred_provider` already existed
and was READ BY NOTHING, while the chevron wrote to the console, so a
provider chosen on one screen was forgotten by the next.

The AI SWITCH stays where it was. Turning the assistant on for this run is a
per-run choice and belongs on the screen; choosing the vendor is not.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def test_no_screen_builds_a_provider_chevron(qtbot, qt_theme_applied):
    """Both screens that had one, checked together.

    Annotate built its own copy of the same control, so a repair applied to
    only one of them is how they drifted apart in the first place.
    """
    from spacr.qt.screens.annotate import AnnotateScreen
    from spacr.qt.screens.app_screen import AppScreen

    for screen in (AppScreen("regression"), AppScreen("mask")):
        qtbot.addWidget(screen)
        assert not hasattr(screen, "_ai_menu_btn"), (
            f"{screen.app_key} grew the provider chevron back")
        assert screen._ai_switch is not None, "the AI switch went with it"

    assert not hasattr(AnnotateScreen, "_refresh_ai_menu"), (
        "Annotate still rebuilds a provider menu")


def test_preferences_offers_the_provider_choice(qtbot, qt_theme_applied):
    """A preference nobody can find is a preference nobody has."""
    from PySide6.QtWidgets import QComboBox, QTabWidget

    from spacr.qt import preferences as prefs

    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    tabs = dialog.findChild(QTabWidget, "PreferencesTabs")
    titles = [tabs.tabText(i) for i in range(tabs.count())]

    assert "AI" in titles, f"no AI tab among {titles}"
    page = dialog.findChild(object, "PreferencesTabAI")
    combos = page.findChildren(QComboBox)

    assert len(combos) == 1, "the AI page should offer exactly one choice"
    # Automatic is first, so a user who has never chosen gets the behaviour
    # the chevron used to give them: the first vendor CLI that is installed.
    assert combos[0].itemData(0) == ""


def test_the_screen_honours_the_preference_when_the_switch_goes_on(
        qtbot, qt_theme_applied, monkeypatch):
    """The preference is read where the chevron used to be obeyed."""
    from spacr.qt import preferences as prefs
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)

    class _Fake:
        name = "vendor-b"
        label = "Vendor B"

    monkeypatch.setattr("spacr.qt.ai.configured_providers",
                        lambda: [_Fake()])
    was = prefs.get_preferred_provider()
    try:
        prefs.set_preferred_provider("vendor-b")
        assert screen._wanted_provider() == "vendor-b"
    finally:
        prefs.set_preferred_provider(was)


def test_a_preference_naming_a_missing_cli_falls_back(qtbot, qt_theme_applied,
                                                      monkeypatch):
    """A preference is a wish, not a guarantee.

    The CLI it names can be uninstalled between sessions, and honouring the
    name regardless would route every question to something that is not
    there -- silently, because the failure is a subprocess that does not
    exist rather than an error the panel can show.
    """
    from spacr.qt import preferences as prefs
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    monkeypatch.setattr("spacr.qt.ai.configured_providers", lambda: [])
    was = prefs.get_preferred_provider()
    try:
        prefs.set_preferred_provider("uninstalled-vendor")
        assert screen._wanted_provider() == ""
    finally:
        prefs.set_preferred_provider(was)
