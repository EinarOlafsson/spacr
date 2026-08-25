"""Walkthroughs on the windows and screens that are missing a part.

``tests/qt/test_walkthrough.py`` drives the real ``MainWindow``: the menu
lists every visible module, an entry runs its own walkthrough, finishing one
marks only that module. What is driven here is everything the derivation has
to survive when the thing it reads is not there -- no app registry, no
settings layout, no Help menu, no screen stack -- because every fact in a
walkthrough is read from somewhere that can be absent.

The stand-in window is deliberate: these are the states a real ``MainWindow``
cannot be put into, and each is a state some other host (a plugin's shell,
the smoke test's stub) really is in.
"""
from __future__ import annotations

import logging
import sys

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QMainWindow, QMenu, QPushButton, QStackedWidget,
                               QWidget)

from spacr.qt import walkthrough as W


@pytest.fixture(autouse=True)
def _clean_seen():
    W.reset()
    yield
    W.reset()


# ---------------------------------------------------------------------------
# The seen flag
# ---------------------------------------------------------------------------

def test_a_flag_written_as_text_still_reads_as_seen():
    """QSettings hands back ``"true"`` from an INI file and ``True`` from a
    native store, and a walkthrough that reran on one platform and not the
    other would be a bug nobody could reproduce."""
    settings = QSettings(W._ORG, W._APP)
    settings.setValue(f"{W._KEY_SEEN}/mask", "true")
    settings.sync()
    assert W.was_seen("mask") is True

    settings.setValue(f"{W._KEY_SEEN}/mask", "no")
    settings.sync()
    assert W.was_seen("mask") is False


# ---------------------------------------------------------------------------
# Deriving the steps
# ---------------------------------------------------------------------------

def test_a_module_the_registry_does_not_know_is_named_by_its_key(monkeypatch):
    monkeypatch.setitem(sys.modules, "spacr.qt.app", None)
    assert W._module_name("not_a_module") == "not_a_module"

    steps = W.build_steps("not_a_module")
    assert steps[0].title == "not_a_module"
    assert "This is the not_a_module module." in steps[0].body


def test_a_key_that_is_in_the_registry_under_no_row_is_still_named():
    """The loop falls through rather than raising on an unknown key."""
    assert W._module_name("no_such_app_key") == "no_such_app_key"


def test_a_layout_that_cannot_be_read_loses_the_settings_beats(monkeypatch,
                                                               caplog):
    """The introduction and the run step do not depend on the form."""
    monkeypatch.setitem(sys.modules, "spacr.qt.screens.settings_model", None)
    with caplog.at_level(logging.DEBUG, logger="spacr.qt.walkthrough"):
        steps = W.build_steps("mask")
    assert "could not read the layout" in caplog.text
    titles = [step.title for step in steps]
    assert "Start at the top" not in titles
    assert titles[0] and titles[-1]


def test_nothing_reads_as_an_empty_sentence():
    assert W._sentence([]) == ""
    assert W._sentence(["one"]) == "one"
    assert W._sentence(["a", "b"]) == "a and b"
    assert W._sentence(["a", "b", "c"]) == "a, b and c"


def test_extra_steps_are_appended_and_can_be_taken_off_again():
    extra = [W.WalkStep(title="Local advice", body="Do it this way.",
                        highlight=None)]
    W.register_steps("mask", extra)
    try:
        assert W.build_steps("mask")[-1].title == "Local advice"
    finally:
        assert W.unregister_steps("mask") is True
    assert W.unregister_steps("mask") is False
    assert W.build_steps("mask")[-1].title != "Local advice"


# ---------------------------------------------------------------------------
# Finding something on the screen to point at
# ---------------------------------------------------------------------------

def test_a_screen_with_no_sections_has_no_first_section(qtbot):
    screen = QWidget()
    qtbot.addWidget(screen)
    assert W._first_section(screen) is None
    assert W._search_bar(screen) is None

    hidden = QWidget()
    qtbot.addWidget(hidden)
    screen._settings_sections = [hidden]
    assert W._first_section(screen) is hidden, \
        "a screen that has never been shown still has a first section"


def test_the_visible_section_wins_over_the_first_one(qtbot):
    screen = QWidget()
    qtbot.addWidget(screen)
    first, second = QWidget(screen), QWidget(screen)
    screen._settings_sections = [first, second]
    screen.show()
    qtbot.waitExposed(screen)
    first.hide()
    assert W._first_section(screen) is second
    screen.hide()


@pytest.mark.parametrize("attr", ["_btn_run", "_run_btn", "_btn_start"])
def test_the_run_button_is_found_by_whichever_name_the_screen_uses(qtbot,
                                                                   attr):
    screen = QWidget()
    qtbot.addWidget(screen)
    button = QPushButton("Go", screen)
    setattr(screen, attr, button)
    assert W._run_button(screen) is button


def test_a_screen_that_names_no_button_is_searched_for_one(qtbot):
    screen = QWidget()
    qtbot.addWidget(screen)
    QPushButton("Cancel", screen)
    run = QPushButton("Run", screen)
    assert W._run_button(screen) is run


def test_a_screen_with_nothing_to_run_points_at_nothing(qtbot):
    screen = QWidget()
    qtbot.addWidget(screen)
    QPushButton("Cancel", screen)
    assert W._run_button(screen) is None


def test_a_highlight_that_cannot_resolve_points_at_nothing(qtbot):
    """The overlay must draw its step even when the widget has gone."""
    screen = QWidget()
    qtbot.addWidget(screen)

    def explode(_screen):
        raise RuntimeError("the widget was deleted")

    bound = W._bind_highlight(explode, screen)
    assert bound(None) is None
    assert W._bind_highlight(None, screen) is None


# ---------------------------------------------------------------------------
# Showing one
# ---------------------------------------------------------------------------

class _Window(QMainWindow):
    """A host with only the seams the walkthrough reaches for."""

    def __init__(self, *, navigate=None, screens=None, stack=None):
        super().__init__()
        self._screens = dict(screens or {})
        if navigate is not None:
            self._on_nav_selected = navigate
        if stack is not None:
            self._stack = stack


def test_a_walkthrough_already_seen_is_not_forced_open(qtbot):
    window = _Window()
    qtbot.addWidget(window)
    W.mark_seen("mask")
    assert W.show_walkthrough(window, "mask", force=False) is None


def test_a_host_that_cannot_navigate_still_gets_its_walkthrough(qtbot,
                                                                caplog):
    """A window with no screen stack is where a plugin shell starts."""
    def refuse(_key):
        raise RuntimeError("no such page")

    window = _Window(navigate=refuse)
    qtbot.addWidget(window)
    with caplog.at_level(logging.DEBUG, logger="spacr.qt.walkthrough"):
        overlay = W.show_walkthrough(window, "mask")
    assert "could not open" in caplog.text
    assert overlay is not None
    qtbot.addWidget(overlay)
    overlay._skip()


def test_a_module_with_no_steps_shows_no_overlay(qtbot, monkeypatch):
    window = _Window()
    qtbot.addWidget(window)
    monkeypatch.setattr(W, "build_steps", lambda key: [])
    assert W.show_walkthrough(window, "mask") is None


# ---------------------------------------------------------------------------
# The wiring
# ---------------------------------------------------------------------------

def test_a_window_with_no_help_menu_gets_no_submenu(qtbot):
    window = _Window()
    qtbot.addWidget(window)
    assert W.install_help_menu(window) is None


def test_the_submenu_lands_at_the_end_when_there_is_no_separator(qtbot,
                                                                  monkeypatch):
    """A Help menu without a separator: the entries still have to arrive."""
    window = _Window()
    qtbot.addWidget(window)
    # Added through the bar, so the QMenu is the bar's child: that is what
    # ``first_run.find_menu`` looks through.
    help_menu = window.menuBar().addMenu("Help")
    help_menu.addAction(QAction("About", help_menu))

    monkeypatch.setattr("spacr.qt.app.APPS", [("mask", "Mask", "Segment", "d")])
    submenu = W.install_help_menu(window)
    assert submenu is not None
    assert help_menu.actions()[-1] is submenu.menuAction()
    assert [a.text() for a in submenu.actions()][:1] == ["Mask"]


def test_a_registry_that_cannot_be_read_still_leaves_a_reset_entry(qtbot,
                                                                   monkeypatch):
    window = _Window()
    qtbot.addWidget(window)
    window.menuBar().addMenu("Help")
    monkeypatch.setitem(sys.modules, "spacr.qt.app", None)

    submenu = W.install_help_menu(window)
    assert [a.text() for a in submenu.actions()] == [
        "Show all walkthroughs again"]


def test_a_window_with_no_stack_keeps_its_handler(qtbot):
    window = _Window()
    qtbot.addWidget(window)
    handler = W.install_window_hooks(window)
    assert handler is not None
    assert getattr(window, "_walkthrough_wired", False) is False
    assert W.install_window_hooks(window) is handler


def test_a_stack_that_refuses_the_connection_is_not_marked_wired(qtbot,
                                                                 monkeypatch,
                                                                 caplog):
    class Refuses:
        def connect(self, _slot):
            raise RuntimeError("wrong thread")

    class Stack:
        currentChanged = Refuses()

    window = _Window(stack=Stack())
    qtbot.addWidget(window)
    with caplog.at_level(logging.DEBUG, logger="spacr.qt.walkthrough"):
        handler = W.install_window_hooks(window)
    assert "could not follow the screen stack" in caplog.text
    assert getattr(window, "_walkthrough_wired", False) is False
    assert handler is not None


def test_a_stack_that_will_not_say_what_is_on_it_offers_nothing(qtbot,
                                                                monkeypatch):
    """Offers nothing is the outcome: a walkthrough shown for a screen
    nobody could name would talk about the wrong module, which raises
    nothing at all."""
    offered = []
    monkeypatch.setattr(W, "maybe_show",
                        lambda window, app_key: offered.append(app_key))

    class Stack:
        def currentWidget(self):
            raise RuntimeError("mid-teardown")

    window = _Window(stack=Stack())
    qtbot.addWidget(window)
    handler = W._WalkthroughHandler(window)
    handler.on_current_changed(0)
    assert offered == []


def test_a_bespoke_screen_is_not_walked_through(qtbot):
    """No shared settings form means no groups to describe."""
    stack = QStackedWidget()
    qtbot.addWidget(stack)
    screen = QWidget()
    screen.app_key = "mask"
    stack.addWidget(screen)
    stack.setCurrentWidget(screen)

    window = _Window(stack=stack)
    qtbot.addWidget(window)
    handler = W._WalkthroughHandler(window)
    handler.on_current_changed(0)
    assert not W.was_seen("mask")


def test_a_screen_that_is_not_a_module_is_not_walked_through(qtbot):
    """Home and the landing page have no app key and nothing to explain."""
    stack = QStackedWidget()
    qtbot.addWidget(stack)
    page = QWidget()
    page._settings_model = object()
    stack.addWidget(page)
    stack.setCurrentWidget(page)

    window = _Window(stack=stack)
    qtbot.addWidget(window)
    W._WalkthroughHandler(window).on_current_changed(0)

    page.app_key = "mask"
    W.mark_seen("mask")
    W._WalkthroughHandler(window).on_current_changed(0)
    assert window.findChildren(W._TourOverlay) == []
