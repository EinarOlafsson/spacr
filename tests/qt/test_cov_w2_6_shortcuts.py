"""A shortcut that cannot fire must not cost the user the window.

Every route out of :mod:`spacr.qt.shortcuts` ends in an optional module --
the command palette, Preferences, recipes, a screen's own search box -- and
each of them is guarded on its own. These drive the real handlers against
real widgets where a widget is what the code touches, and inject the failure
where the point is what happens when an optional piece is missing.
"""
from __future__ import annotations

import logging

import pytest
from PySide6.QtCore import QEvent, QSize
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import QLabel, QMainWindow, QWidget

from spacr.qt import shortcuts as sc


@pytest.fixture
def window(qapp):
    win = QMainWindow()
    win.resize(900, 600)
    yield win
    win.close()
    win.deleteLater()


# --------------------------------------------------------------------------
# the declared map
# --------------------------------------------------------------------------

def test_a_key_is_printed_in_the_spelling_the_keyboard_has():
    """Writing "Ctrl+H" into a label hard-codes one platform into the help."""
    assert sc.native("Ctrl+H") == QKeySequence("Ctrl+H").toString(
        QKeySequence.NativeText)


def test_a_key_qt_cannot_parse_is_shown_as_written(monkeypatch):
    """A binding added at runtime may be spelled in a way Qt refuses; the
    cheat sheet still has to name it rather than show a blank row."""
    def _refuse(_keys):
        raise ValueError("not a key sequence")

    monkeypatch.setattr(sc, "QKeySequence", _refuse)
    assert sc.native("Mod4+Space") == "Mod4+Space"


def test_the_map_is_the_window_wide_keys_plus_the_per_screen_ones():
    """The distinction is real -- one set is always live and the other is
    not -- and the map has to describe both."""
    everything = sc.mapped()
    assert everything[:len(sc.SHORTCUTS)] == sc.SHORTCUTS
    assert set(sc.SCREEN_SHORTCUTS).issubset(set(everything))
    assert all(spec.scope != sc.EVERYWHERE for spec in sc.SCREEN_SHORTCUTS)


def test_install_binds_every_key_it_is_responsible_for(window):
    """`installed()` is the promise that `install()` wires these; a key it
    lists and nobody binds on the window is a cheat-sheet entry that does
    nothing where the sheet says it works.

    It held an expected failure for `Ctrl+End`, which `installed()` claimed
    while the only holder was the console panel's own -- built with the
    screen, inert while that panel is hidden, and absent from a fresh
    window. `install()` binds it now, so the promise is kept for every key
    and the expected failure is gone.
    """
    sc.install(window)
    bound = {shortcut.key().toString(QKeySequence.NativeText)
             for shortcut in window.findChildren(QShortcut)}
    for spec in sc.installed():
        assert sc.native(spec.keys) in bound, spec.keys


def test_install_binds_every_key_that_is_not_bound_elsewhere(window):
    sc.install(window)
    bound = {shortcut.key().toString(QKeySequence.NativeText)
             for shortcut in window.findChildren(QShortcut)}
    for spec in sc.installed():
        if spec.scope != sc.EVERYWHERE:
            continue
        assert sc.native(spec.keys) in bound, spec.keys
    # Window actions own these rather than ``install()``.
    for keys in sc.BOUND_ELSEWHERE:
        assert sc.native(keys) not in bound


# --------------------------------------------------------------------------
# discovering what is live
# --------------------------------------------------------------------------

def test_a_shortcut_added_at_runtime_appears_without_a_list_being_edited(
        window):
    QShortcut(QKeySequence("Ctrl+Alt+Z"), window)
    found = sc.discover(window)
    assert any(spec.keys == sc.native("Ctrl+Alt+Z") for spec in found)
    assert all(spec.category == "Other" for spec in found)


def test_a_menu_action_contributes_its_own_text_as_the_description(window):
    action = QAction("&Export figure", window)
    action.setShortcut(QKeySequence("Ctrl+Alt+E"))
    window.addAction(action)
    found = {spec.keys: spec.label for spec in sc.discover(window)}
    assert found[sc.native("Ctrl+Alt+E")] == "Export figure"


def test_a_declared_key_is_left_to_its_declaration(window):
    """That is where the label and the scope live; discovering it again
    would list it twice with "(not described)" beside it."""
    QShortcut(QKeySequence("Ctrl+H"), window)
    assert not [spec for spec in sc.discover(window)
                if spec.keys == sc.native("Ctrl+H")]


def test_a_window_that_cannot_be_searched_discovers_nothing():
    class Hostile:
        def findChildren(self, *_args):
            raise RuntimeError("the window is already gone")

    assert sc.discover(Hostile()) == []


def test_a_holder_that_cannot_name_its_key_is_skipped_not_fatal(window):
    """One unreadable action must not empty the whole cheat sheet."""
    class Broken:
        def shortcut(self):
            raise RuntimeError("deleted underneath us")

    class Mixed:
        def findChildren(self, kind):
            return [Broken()] if kind is not QShortcut else []

    assert sc.discover(Mixed()) == []


def test_the_same_key_on_two_holders_is_listed_once(window):
    QShortcut(QKeySequence("Ctrl+Alt+Y"), window)
    QShortcut(QKeySequence("Ctrl+Alt+Y"), window)
    keys = [spec.keys for spec in sc.discover(window)]
    assert keys.count(sc.native("Ctrl+Alt+Y")) == 1


# --------------------------------------------------------------------------
# the window hooks
# --------------------------------------------------------------------------

#: Every optional module ``_install_window_hooks`` wires, in the order it
#: reaches them.
HOOK_MODULES = [
    "spacr.qt.widgets.feature_dictionary",
    "spacr.qt.settings_search",
    "spacr.qt.recipes",
    "spacr.qt.preview_registry",
    "spacr.qt.walkthrough",
]


@pytest.mark.parametrize("broken", HOOK_MODULES)
def test_an_optional_hook_that_fails_never_costs_anyone_a_window(
        window, monkeypatch, broken):
    """The failing hook is stepped over, not stopped at.

    Every other hook still runs and so does the menu-role sweep that
    follows them all -- which is the difference that matters, because
    "nothing escaped" is equally true of a guard that ran no hook at all.
    """
    import importlib

    ran = []
    for path in HOOK_MODULES:
        module = importlib.import_module(path)
        if path == broken:
            def _explode(_window, _path=path):
                ran.append(_path)
                raise RuntimeError(f"{_path} is unhappy")
            monkeypatch.setattr(module, "install_window_hooks", _explode)
        else:
            monkeypatch.setattr(
                module, "install_window_hooks",
                lambda _window, _path=path: ran.append(_path))
    swept = []
    window.pin_all_menu_roles = lambda: swept.append(True)

    sc._install_window_hooks(window)

    assert ran == HOOK_MODULES
    assert swept == [True]


def test_the_macos_menu_roles_are_re_pinned_after_the_hooks_run(window):
    """Everything above may have added menu-bar actions, and an action with
    no explicit role is one Qt assigns from its TEXT."""
    calls = []
    window.pin_all_menu_roles = lambda: calls.append(True)
    sc._install_window_hooks(window)
    assert calls == [True]


def test_a_menu_role_sweep_that_fails_is_not_fatal_either(window):
    """The sweep is the last thing the function does, so a swallowed failure
    and a sweep that never ran look identical from outside; the call records
    itself before it throws so the two can be told apart."""
    calls = []

    def _explode():
        calls.append(True)
        raise RuntimeError("no menu bar")

    window.pin_all_menu_roles = _explode
    sc._install_window_hooks(window)
    assert calls == [True]


# --------------------------------------------------------------------------
# what each key actually does
# --------------------------------------------------------------------------

class _FakeStackWindow:
    """A window-shaped object holding one screen, as `_stack` exposes it."""

    def __init__(self, screen):
        self._stack = type("Stack", (), {
            "currentWidget": staticmethod(lambda: screen)})()


def test_a_screen_that_wants_the_question_mark_keeps_it(monkeypatch):
    """`?` opening a modal sheet over a rapid-labelling session is exactly
    the wrong response, so a screen may claim the key."""
    shown = []
    monkeypatch.setattr(sc, "show_cheat_sheet", lambda parent: shown.append(1))

    class Annotate:
        def handle_key(self, key):
            return key == "?"

    sc._help_key(_FakeStackWindow(Annotate()))
    assert shown == []


def test_a_screen_that_declines_the_question_mark_gets_the_cheat_sheet(
        monkeypatch):
    shown = []
    monkeypatch.setattr(sc, "show_cheat_sheet", lambda parent: shown.append(1))

    class Plain:
        def handle_key(self, key):
            return False

    sc._help_key(_FakeStackWindow(Plain()))
    assert shown == [1]


def test_a_screen_whose_key_handler_raises_still_gets_the_cheat_sheet(
        monkeypatch):
    shown = []
    monkeypatch.setattr(sc, "show_cheat_sheet", lambda parent: shown.append(1))

    class Broken:
        def handle_key(self, key):
            raise RuntimeError("mid-repaint")

    sc._help_key(_FakeStackWindow(Broken()))
    assert shown == [1]


def test_a_window_with_no_stack_still_answers_the_question_mark(monkeypatch):
    shown = []
    monkeypatch.setattr(sc, "show_cheat_sheet", lambda parent: shown.append(1))
    sc._help_key(object())
    assert shown == [1]


def test_a_numbered_key_navigates_to_that_visible_app(monkeypatch):
    from spacr.qt import app as qt_app

    monkeypatch.setattr(qt_app, "APPS", [("mask", "Mask"), ("measure", "M")],
                        raising=False)
    monkeypatch.setattr(qt_app, "app_is_visible", lambda key: True,
                        raising=False)
    picked = []
    window = type("W", (), {"_on_nav_selected": lambda self, key:
                            picked.append(key)})()
    sc._nav_by_index(window, 1)
    assert picked == ["measure"]


def test_a_numbered_key_past_the_end_of_the_sidebar_does_nothing(monkeypatch):
    from spacr.qt import app as qt_app

    monkeypatch.setattr(qt_app, "APPS", [("mask", "Mask")], raising=False)
    monkeypatch.setattr(qt_app, "app_is_visible", lambda key: True,
                        raising=False)
    picked = []
    window = type("W", (), {"_on_nav_selected": lambda self, key:
                            picked.append(key)})()
    sc._nav_by_index(window, 8)
    assert picked == []


def test_a_sidebar_that_cannot_be_consulted_swallows_the_number_key(
        monkeypatch):
    """Ctrl+3 during teardown must not raise out of a global shortcut."""
    from spacr.qt import app as qt_app

    monkeypatch.setattr(qt_app, "APPS", [("mask", "Mask")], raising=False)

    def _explode(_key):
        raise RuntimeError("the registry is gone")

    monkeypatch.setattr(qt_app, "app_is_visible", _explode, raising=False)
    picked = []
    window = type("W", (), {"_on_nav_selected": lambda self, key:
                            picked.append(key)})()
    sc._nav_by_index(window, 0)
    assert picked == []


def test_a_hidden_app_is_not_reachable_by_its_number(monkeypatch):
    from spacr.qt import app as qt_app

    monkeypatch.setattr(qt_app, "APPS", [("mask", "Mask")], raising=False)
    monkeypatch.setattr(qt_app, "app_is_visible", lambda key: False,
                        raising=False)
    picked = []
    window = type("W", (), {"_on_nav_selected": lambda self, key:
                            picked.append(key)})()
    sc._nav_by_index(window, 0)
    assert picked == []


def test_a_navigation_key_on_a_window_without_the_route_is_ignored():
    """The missing route has to be the reason nothing happened, so the
    lookup is recorded: a guard that bailed earlier would never have asked
    for `_on_nav_selected` at all."""
    asked = []

    class NoRoute:
        def __getattr__(self, name):
            asked.append(name)
            raise AttributeError(name)

    sc._nav(NoRoute(), "__home__")
    assert asked == ["_on_nav_selected"]


def test_the_palette_key_opens_the_palette(monkeypatch, window):
    from spacr.qt import command_palette

    opened = []

    class FakePalette:
        def __init__(self, parent):
            opened.append(parent)

        def exec(self):
            return 0

    monkeypatch.setattr(command_palette, "CommandPalette", FakePalette)
    sc._open_palette(window)
    assert opened == [window]


def test_a_palette_that_cannot_be_built_is_not_a_crash(monkeypatch, window,
                                                       caplog):
    """The palette was really reached -- a guard that never tried also does
    not raise -- and what went wrong is left in the log rather than lost."""
    from spacr.qt import command_palette

    tried = []

    def _explode(parent):
        tried.append(parent)
        raise RuntimeError("no registry")

    monkeypatch.setattr(command_palette, "CommandPalette", _explode)
    with caplog.at_level(logging.DEBUG, logger="spacr.qt.shortcuts"):
        sc._open_palette(window)
    assert tried == [window]
    assert "no registry" in caplog.text


def test_the_preferences_key_opens_preferences(monkeypatch, window):
    from spacr.qt import preferences

    opened = []

    class FakeDialog:
        def __init__(self, parent):
            opened.append(parent)

        def exec(self):
            return 0

    monkeypatch.setattr(preferences, "PreferencesDialog", FakeDialog)
    sc._open_preferences(window)
    assert opened == [window]


def test_preferences_that_cannot_be_built_is_not_a_crash(monkeypatch, window,
                                                        caplog):
    """As with the palette: the dialog was attempted, and the reason it
    could not be built survives in the log."""
    from spacr.qt import preferences

    tried = []

    def _explode(parent):
        tried.append(parent)
        raise RuntimeError("no settings store")

    monkeypatch.setattr(preferences, "PreferencesDialog", _explode)
    with caplog.at_level(logging.DEBUG, logger="spacr.qt.shortcuts"):
        sc._open_preferences(window)
    assert tried == [window]
    assert "no settings store" in caplog.text


def test_the_recipes_key_opens_the_recipe_menu(monkeypatch, window):
    from spacr.qt import recipes

    fired = []

    class FakeHandler:
        def __init__(self, parent):
            self.parent = parent

        def on_triggered(self):
            fired.append(self.parent)

    monkeypatch.setattr(recipes, "_RecipeMenuHandler", FakeHandler)
    sc._open_recipes(window)
    assert fired == [window]


def test_a_recipe_dialog_that_is_unavailable_is_not_a_crash(monkeypatch,
                                                            window, caplog):
    """The handler was constructed against this window before it refused,
    and the refusal is logged rather than silently dropped."""
    from spacr.qt import recipes

    tried = []

    def _explode(parent):
        tried.append(parent)
        raise RuntimeError("no module on screen")

    monkeypatch.setattr(recipes, "_RecipeMenuHandler", _explode)
    with caplog.at_level(logging.DEBUG, logger="spacr.qt.shortcuts"):
        sc._open_recipes(window)
    assert tried == [window]
    assert "no module on screen" in caplog.text


def test_the_ai_key_toggles_the_switch_on_the_visible_screen(monkeypatch):
    from spacr.qt.screens import app_screen

    class FakeSwitch:
        def __init__(self):
            self.state = False

        def isChecked(self):
            return self.state

        def setChecked(self, value):
            self.state = value

    class FakeScreen:
        def __init__(self, visible):
            self._visible = visible
            self._ai_switch = FakeSwitch()

        def isVisible(self):
            return self._visible

    hidden, shown = FakeScreen(False), FakeScreen(True)
    window = type("W", (), {"findChildren": lambda self, kind:
                            [hidden, shown]})()
    monkeypatch.setattr(app_screen, "AppScreen", object, raising=False)
    sc._toggle_ai(window)
    assert shown._ai_switch.isChecked() is True
    assert hidden._ai_switch.isChecked() is False


def test_the_ai_key_on_a_window_that_cannot_be_searched_is_logged(caplog):
    """The sweep was attempted, and for the one type that carries the
    switch: a handler that searched some other class would find a screen
    without `_ai_switch` and toggle nothing while raising nothing either.

    The failure remains non-fatal, but its traceback must reach the debug log
    so a broken advertised shortcut can be diagnosed from a user's log.
    """
    from spacr.qt.screens.app_screen import AppScreen

    searched = []

    class Hostile:
        def findChildren(self, kind):
            searched.append(kind)
            raise RuntimeError("window is gone")

    with caplog.at_level(logging.DEBUG, logger="spacr.qt.shortcuts"):
        sc._toggle_ai(Hostile())

    assert searched == [AppScreen]
    record = next(record for record in caplog.records
                  if record.getMessage() == "could not toggle the AI switch")
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError


# --------------------------------------------------------------------------
# Ctrl+F -- find a setting
# --------------------------------------------------------------------------

def test_ctrl_f_puts_the_caret_in_the_settings_search_box(qapp):
    from PySide6.QtWidgets import QLineEdit

    entry = QLineEdit()
    entry.setText("cell diameter")
    bar = type("Bar", (), {})()
    bar._input = entry
    screen = type("Screen", (), {})()
    screen._settings_search = bar
    sc._focus_settings_search(_FakeStackWindow(screen))
    assert entry.selectedText() == "cell diameter"


def test_a_screen_with_no_search_strip_is_left_alone_not_swallowed():
    """Left alone means exactly one question asked of the screen -- does it
    have a search strip -- and nothing touched once the answer is no."""
    asked = []

    class Screen:
        def __getattr__(self, name):
            asked.append(name)
            raise AttributeError(name)

    sc._focus_settings_search(_FakeStackWindow(Screen()))
    assert asked == ["_settings_search"]


def test_a_window_with_no_stack_cannot_be_searched():
    """It stops at the missing stack rather than going on to hunt for a
    search strip on whatever `currentWidget` did not return."""
    asked = []

    class NoStack:
        def __getattr__(self, name):
            asked.append(name)
            raise AttributeError(name)

    sc._focus_settings_search(NoStack())
    assert asked == ["_stack"]


def test_a_search_box_that_refuses_focus_is_logged_not_raised(caplog):
    """Logged is the whole claim: the box was really asked for the caret,
    and the refusal reaches the log instead of disappearing."""
    tried = []

    class Refusing:
        def setFocus(self):
            tried.append("setFocus")
            raise RuntimeError("already deleted")

    bar = type("Bar", (), {})()
    bar._input = Refusing()
    screen = type("Screen", (), {})()
    screen._settings_search = bar
    with caplog.at_level(logging.DEBUG, logger="spacr.qt.shortcuts"):
        sc._focus_settings_search(_FakeStackWindow(screen))
    assert tried == ["setFocus"]
    assert "settings search" in caplog.text


# --------------------------------------------------------------------------
# the overlay
# --------------------------------------------------------------------------

def test_the_overlay_lists_every_key_the_map_describes(window):
    overlay = sc.show_cheat_sheet(window)
    assert overlay is not None
    printed = {label.text() for label in overlay.findChildren(QLabel)}
    for spec in sc.mapped():
        assert sc.native(spec.keys) in printed, spec.keys
    assert "Press any key to close." in printed
    overlay.dismiss()


def test_a_scoped_key_says_where_it_works(window):
    """A key that works on one screen and is listed without saying so sends
    a user to press it somewhere it does nothing."""
    overlay = sc.show_cheat_sheet(window)
    printed = {label.text() for label in overlay.findChildren(QLabel)}
    scoped = [spec for spec in sc.mapped() if spec.scope != sc.EVERYWHERE]
    assert scoped
    for spec in scoped:
        assert f"{spec.label}  —  {spec.scope}" in printed
    overlay.dismiss()


def test_the_card_never_grows_wider_than_the_window_it_covers(window):
    """A map that runs off the screen is the same fault as a map that leaves
    keys out."""
    window.resize(1280, 800)
    overlay = sc.show_cheat_sheet(window)
    assert overlay._card.width() <= overlay.width()
    overlay.dismiss()


def test_a_narrow_window_wraps_the_categories_instead_of_widening(window):
    window.resize(700, 600)
    narrow = sc.show_cheat_sheet(window)
    narrow_rows = narrow._card_content.layout().rowCount()
    narrow.dismiss()
    window.resize(1900, 900)
    wide = sc.show_cheat_sheet(window)
    assert narrow_rows > wide._card_content.layout().rowCount()
    wide.dismiss()


def test_the_overlay_dims_whatever_is_behind_the_card(window, qapp):
    """That is what makes it an answer laid over the window rather than a
    mode the user has to leave."""
    from PySide6.QtGui import QColor

    overlay = sc.show_cheat_sheet(window)
    qapp.processEvents()
    shot = overlay.grab().toImage()
    corner = QColor(shot.pixel(2, overlay.height() - 3))
    assert corner.red() < 120 and corner.green() < 120 and corner.blue() < 120
    overlay.dismiss()


def test_the_card_stays_centred_when_the_window_resizes(window, qapp):
    overlay = sc.show_cheat_sheet(window)
    qapp.processEvents()
    overlay.setGeometry(0, 0, 1400, 1600)
    overlay.resizeEvent(None)
    card = overlay._card.geometry()
    assert abs(card.center().x() - 700) <= 1
    assert abs(card.center().y() - 800) <= 1
    overlay.dismiss()


def test_any_key_closes_the_overlay(window, qapp):
    overlay = sc.show_cheat_sheet(window)
    overlay.keyPressEvent(None)
    qapp.processEvents()
    assert not overlay.isVisible()


def test_a_click_anywhere_closes_it_too(window, qapp):
    overlay = sc.show_cheat_sheet(window)
    overlay.mousePressEvent(None)
    qapp.processEvents()
    assert not overlay.isVisible()


def test_the_overlay_follows_the_window_when_it_is_resized(window, qapp):
    overlay = sc.show_cheat_sheet(window)
    window.resize(1100, 700)
    qapp.processEvents()
    overlay.eventFilter(window, QEvent(QEvent.Resize))
    assert overlay.size() == window.rect().size()
    overlay.dismiss()


def test_re_showing_the_sheet_replaces_the_overlay_rather_than_stacking(
        window, qapp):
    first = sc.show_cheat_sheet(window)
    second = sc.show_cheat_sheet(window)
    qapp.processEvents()
    assert second is not first
    assert window._spacr_shortcut_overlay is second
    second.dismiss()


def test_an_overlay_whose_window_is_already_gone_dismisses_quietly(window,
                                                                   qapp):
    """Detaching the filter is the first half of dismissal; failing it must
    not cost the second, or the overlay stays up over a dead window and is
    never freed."""
    from shiboken6 import isValid

    overlay = sc.show_cheat_sheet(window)
    overlay.show()
    assert not overlay.isHidden()

    class Gone:
        def removeEventFilter(self, _filter):
            raise RuntimeError("wrapped C/C++ object has been deleted")

    overlay._window = Gone()
    overlay.dismiss()
    assert overlay.isHidden()
    qapp.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not isValid(overlay)


def test_a_stale_overlay_that_cannot_be_dismissed_is_replaced_anyway(window):
    class Stale:
        def dismiss(self):
            raise RuntimeError("wrapped C/C++ object has been deleted")

    window._spacr_shortcut_overlay = Stale()
    overlay = sc.show_cheat_sheet(window)
    assert overlay is not None
    overlay.dismiss()


def test_a_parentless_caller_gets_a_dialog_rather_than_an_overlay(monkeypatch,
                                                                  qapp):
    """An overlay over nothing would have nothing to cover."""
    from PySide6.QtWidgets import QDialog

    monkeypatch.setattr(QDialog, "exec", lambda self: 0)
    assert sc.show_cheat_sheet(None) is None


def test_a_zero_sized_window_gets_the_dialog_too(monkeypatch, qapp):
    from PySide6.QtWidgets import QDialog

    monkeypatch.setattr(QDialog, "exec", lambda self: 0)
    tiny = QWidget()
    tiny.resize(QSize(0, 0))
    try:
        assert sc.show_cheat_sheet(tiny) is None
    finally:
        tiny.deleteLater()


def test_the_overlay_qss_names_the_card_the_theme_has_to_reach():
    palette = {"theme": "dark", "accent": "#00aaff", "fg": "#eeeeee",
               "fg_dim": "#999999"}
    qss = sc._overlay_qss(palette, 1.0)
    assert f"QWidget#{sc.OVERLAY_CARD_NAME}" in qss
    assert "#00aaff" in qss
