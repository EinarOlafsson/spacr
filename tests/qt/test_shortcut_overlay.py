"""The ``?`` overlay, and a command palette that reaches settings.

Pressing ``?`` asks "what can I press here" and the answer is worth about
two seconds. A modal dialog makes the user commit to a mode, find a close
button and leave it; an overlay answers and gets out of the way on the next
keystroke. Both halves are asserted against rendered geometry, because "the
card is centred and fits" is the whole difference between an overlay and a
rectangle of text over the middle of somebody's work.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QKeyEvent, QKeySequence
from PySide6.QtWidgets import QLabel, QWidget

from spacr.qt.shortcuts import (
    SHORTCUTS,
    ShortcutOverlay,
    _bind,
    install,
    show_cheat_sheet,
)


@pytest.fixture
def window(qtbot):
    from spacr.qt.app import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1280, 860)
    win.show()
    qtbot.waitExposed(win)
    return win


# ---------------------------------------------------------------------------
# 1. The overlay
# ---------------------------------------------------------------------------

def test_the_question_mark_gives_an_overlay_not_a_dialog(window, qtbot):
    overlay = show_cheat_sheet(window)
    qtbot.addWidget(overlay)
    assert isinstance(overlay, ShortcutOverlay)
    assert overlay.parentWidget() is window
    overlay.dismiss()


def test_the_overlay_covers_the_window(window, qtbot):
    overlay = show_cheat_sheet(window)
    qtbot.addWidget(overlay)
    assert overlay.size() == window.rect().size()
    overlay.dismiss()


def test_the_card_is_centred_and_fits(window, qtbot):
    """Rendered geometry, not a promise about a layout."""
    overlay = show_cheat_sheet(window)
    qtbot.addWidget(overlay)
    overlay.show()
    qtbot.waitExposed(overlay)

    card = overlay._card
    assert card.width() > 0 and card.height() > 0
    assert card.width() <= overlay.width()
    assert card.height() <= overlay.height()

    card_centre = card.geometry().center()
    overlay_centre = overlay.rect().center()
    assert abs(card_centre.x() - overlay_centre.x()) <= 1
    assert abs(card_centre.y() - overlay_centre.y()) <= 1
    overlay.dismiss()


def test_the_recrop_row_remains_reachable_in_a_short_window(window, qtbot):
    """Growing the complete map may scroll its inside, never clip its tail."""
    overlay = show_cheat_sheet(window)
    qtbot.addWidget(overlay)
    overlay.show()
    qtbot.waitExposed(overlay)

    recrop = next(
        label
        for label in overlay._card.findChildren(QLabel)
        if label.text().startswith("Recrop an object")
    )
    overlay._scroll.ensureWidgetVisible(recrop)
    qtbot.wait(10)

    top = recrop.mapTo(overlay._scroll.viewport(), QPoint()).y()
    assert top < overlay._scroll.viewport().height()
    assert top + recrop.height() > 0
    assert overlay._card.height() <= overlay.height()
    overlay.dismiss()


def test_every_registered_shortcut_is_on_the_card(window, qtbot):
    """EVERY MAPPED ONE, not only the window's own (197).

    `SHORTCUTS` is what `install()` binds; `SCREEN_SHORTCUTS` is what the
    screens bind and the map still has to describe. `mapped()` is both.
    """
    from spacr.qt.shortcuts import mapped, native

    overlay = show_cheat_sheet(window)
    qtbot.addWidget(overlay)
    rendered = {label.text() for label in overlay._card.findChildren(QLabel)}
    for spec in mapped():
        # PRINTED IN THE PLATFORM'S SPELLING, so the comparison is against
        # what the card actually shows rather than Qt's portable form.
        assert native(spec.keys) in rendered, \
            f"{spec.keys} is bound but not shown"
        # A ROW STARTS WITH THE LABEL and may continue with the scope --
        # "Brush  —  the Make Masks screen" -- because a key that works on
        # one screen and is listed without saying so sends a user to press
        # it somewhere it does nothing.
        assert any(text.startswith(spec.label) for text in rendered), \
            f"{spec.label} is missing its row"
    overlay.dismiss()


def test_a_per_screen_row_says_where_it_works(window, qtbot):
    overlay = show_cheat_sheet(window)
    qtbot.addWidget(overlay)
    rendered = {label.text() for label in overlay._card.findChildren(QLabel)}

    assert any("Make Masks screen" in text for text in rendered)
    overlay.dismiss()


def test_the_categories_are_laid_out_in_columns(window, qtbot):
    """Fifteen bindings in one column is a scroll; in three it is a glance.

    TILED, NOT ONE PAIR PER CATEGORY. The map grew from 17 rows to 33 (197)
    and a column-pair for every category made the card 1,640 px wide against
    a 1,280 px overlay -- a map that runs off the screen is the same fault
    as one that leaves keys out. Categories fill the width and then wrap, so
    what is asserted is that they SHARE ROWS, not that every one has a
    column of its own.
    """
    overlay = show_cheat_sheet(window)
    qtbot.addWidget(overlay)
    overlay.show()
    qtbot.waitExposed(overlay)
    headers = [lbl for lbl in overlay._card.findChildren(QLabel)
               if lbl.objectName() == "ShortcutOverlayCategory"]
    assert len(headers) >= 2
    xs = {lbl.x() for lbl in headers}
    assert len(xs) > 1, "the categories stacked instead of tiling"


def test_the_card_never_runs_off_the_overlay(window, qtbot):
    """The reason the layout wraps at all."""
    overlay = show_cheat_sheet(window)
    qtbot.addWidget(overlay)
    overlay.show()
    qtbot.waitExposed(overlay)

    assert overlay._card.width() <= overlay.width()
    overlay.dismiss()


def test_any_key_closes_it(window, qtbot):
    overlay = show_cheat_sheet(window)
    qtbot.addWidget(overlay)
    overlay.keyPressEvent(
        QKeyEvent(QKeyEvent.KeyPress, Qt.Key_A, Qt.NoModifier, "a"))
    assert not overlay.isVisible()


def test_asking_twice_leaves_one_overlay(window, qtbot):
    first = show_cheat_sheet(window)
    qtbot.addWidget(first)
    second = show_cheat_sheet(window)
    qtbot.addWidget(second)
    assert second is not first
    assert not first.isVisible()
    live = [o for o in window.findChildren(ShortcutOverlay) if o.isVisible()]
    assert len(live) == 1
    second.dismiss()


def test_a_sizeless_parent_still_gets_the_dialog(qtbot, monkeypatch):
    """The fallback exists for a parentless or zero-sized caller, where an
    overlay would have nothing to cover."""
    bare = QWidget()
    qtbot.addWidget(bare)
    bare.resize(0, 0)
    shown = []
    monkeypatch.setattr(
        "PySide6.QtWidgets.QDialog.exec", lambda self: shown.append(self))
    assert show_cheat_sheet(bare) is None
    assert shown


def test_the_new_bindings_are_both_declared_and_wired(window):
    """A binding documented but not bound, or bound but not documented, is
    the failure `SHORTCUTS` exists to prevent."""
    declared = {s.keys for s in SHORTCUTS}
    assert {"Ctrl+F", "Ctrl+Shift+R"} <= declared

    # A QAction'S SHORTCUT IS BOUND TOO. `Ctrl+Shift+A` opens the full app
    # list and is set on a window action rather than on a QShortcut -- looking
    # only at QShortcut called a wired key unwired. The rule this test is
    # for is "documented and reachable", and either kind reaches.
    from PySide6.QtGui import QAction, QShortcut

    bound = {sc.key().toString() for sc in window.findChildren(QShortcut)}
    bound |= {a.shortcut().toString() for a in window.findChildren(QAction)
              if not a.shortcut().isEmpty()}
    for keys in declared:
        assert keys in bound, f"{keys} is on the cheat sheet but not bound"


def test_two_live_windows_do_not_make_their_shortcuts_ambiguous(qtbot):
    """A second spaCR window has its own keys, not a competing app-global
    copy of the first window's keys.

    Qt suppresses both callbacks when two ``ApplicationShortcut`` objects
    carry one sequence.  That used to make Ctrl+End intermittent after an
    old window survived until deferred deletion; driving two live windows is
    the order-independent regression for that failure.
    """
    from PySide6.QtWidgets import QMainWindow

    first = QMainWindow()
    second = QMainWindow()
    qtbot.addWidget(first)
    qtbot.addWidget(second)
    first.resize(320, 200)
    second.resize(320, 200)
    first.show()
    second.show()

    fired = []
    one = _bind(first, "Ctrl+Alt+9", lambda: fired.append("first"))
    two = _bind(second, "Ctrl+Alt+9", lambda: fired.append("second"))
    assert one.context() == two.context() == Qt.WindowShortcut

    second.raise_()
    second.activateWindow()
    qtbot.waitUntil(second.isActiveWindow, timeout=2000)
    qtbot.keyClick(second, Qt.Key_9, Qt.ControlModifier | Qt.AltModifier)

    assert fired == ["second"]


def _open_console(window, qtbot):
    """The Mask screen's console, unfolded and on screen.

    The console is foldable and a folded one is `isHidden()`, which is
    exactly the state in which its own `Ctrl+End` could never fire: a
    `Qt.WindowShortcut` whose parent widget is hidden is not active.
    """
    window._on_nav_selected("mask")
    qtbot.wait(50)
    screen = window._screens["mask"]
    screen._console_folder.set_shut(False)
    qtbot.wait(20)
    console = screen._console
    assert console.isVisible(), "the console never reached the screen"
    return console


def test_ctrl_end_really_sends_the_console_to_its_newest_line(window, qtbot):
    """DRIVEN, not looked up in a table.

    `Ctrl+End` was declared three times and bound nowhere `installed()` could
    see it: the only holder was the console panel's own, which does not
    exist until a module screen is built and is inert while that panel is
    hidden. So a fresh window -- the one a user reads the cheat sheet on --
    carried no `Ctrl+End` at all.

    THE WINDOW'S BINDING IS THE ONE UNDER TEST. The panel's copy stands down
    (two live holders of one key is ambiguous, and an ambiguous shortcut
    fires neither), so it is asserted inert BEFORE the key is pressed:
    whatever moves the scrollbar below can only be the window's.
    """
    console = _open_console(window, qtbot)
    _activate(window, qtbot)
    qtbot.wait(50)
    assert not console._end_shortcut.isEnabled(), \
        "the panel still holds Ctrl+End; this would not be the window's jump"
    # The range is given to the bar rather than grown out of console output,
    # for the reason the console's own tests give: headless, the entries lay
    # out to nothing and the bar's maximum is 0, so EVERY position is the
    # end and a jump could not be told from doing nothing. It is set after
    # the window is activated, because that runs a layout pass which
    # recomputes the range away -- and asserted below, so a range that goes
    # missing fails the test rather than quietly emptying it.
    bar = console._scroll.verticalScrollBar()
    bar.setRange(0, 2000)
    bar.setValue(0)
    console._follow_output = False
    assert bar.maximum() > 0 and bar.value() == 0

    qtbot.keyClick(window, Qt.Key_End, Qt.ControlModifier)

    assert bar.maximum() > 0, \
        "the scrollbar lost its range; nothing here was proved"
    assert bar.value() == bar.maximum()
    # Both halves of the jump, because they are one decision: a console that
    # jumped without resuming the follow would slide off the end on the very
    # next line written.
    assert console._follow_output


def test_ctrl_end_is_harmless_on_a_screen_with_no_console(window, qtbot):
    """Home has none, and "harmless" is more than "it did not raise": the
    screen is left exactly as it was found, rather than the key navigating
    somewhere or taking the caret off whatever had it."""
    window._on_nav_selected("__home__")
    qtbot.wait(20)
    _activate(window, qtbot)
    home = window._stack.currentWidget()
    before = window.focusWidget()

    qtbot.keyClick(window, Qt.Key_End, Qt.ControlModifier)

    assert window._stack.currentWidget() is home
    assert window.focusWidget() is before


def test_only_one_ctrl_end_is_live_at_a_time(window, qtbot):
    """Two holders of one key is Qt's definition of AMBIGUOUS, and an
    ambiguous shortcut fires NEITHER handler -- measured: with the window's
    binding and the console's own both live, `activated` stays silent on
    both. The window's is the one that reaches every screen, so the panel's
    copy stands down as the screen is shown.
    """
    from PySide6.QtGui import QShortcut

    console = _open_console(window, qtbot)
    live = [sc for sc in window.findChildren(QShortcut)
            if sc.key() == QKeySequence("Ctrl+End") and sc.isEnabled()]
    assert len(live) == 1, [sc.parentWidget() for sc in live]
    assert console._end_shortcut in window.findChildren(QShortcut)
    assert not console._end_shortcut.isEnabled()


def test_the_cheat_sheet_describes_ctrl_end_once(window, qtbot):
    """It was declared three times -- a prose block and two `ShortcutSpec`
    entries whose descriptions disagreed ("Jump to the newest console line"
    against "Jump to the newest line") -- so the map printed the same key
    twice, under two categories, saying two things.
    """
    from spacr.qt.shortcuts import mapped, native

    ends = [spec for spec in mapped() if spec.keys == "Ctrl+End"]
    assert len(ends) == 1, [spec.label for spec in ends]

    overlay = show_cheat_sheet(window)
    qtbot.addWidget(overlay)
    printed = [label.text() for label in overlay._card.findChildren(QLabel)]
    assert printed.count(native("Ctrl+End")) == 1, printed
    overlay.dismiss()


def _activate(window, qtbot) -> None:
    """Make ``window`` the active window, and prove it took.

    ``QWidget.hasFocus()`` is False for every widget whose top-level window
    is not ACTIVE, so a focus assertion in an inactive window measures
    activation instead of focus. Under ``QT_QPA_PLATFORM=offscreen`` there
    is no window manager to hand activation out: ``show()`` never activates,
    and — measured, not assumed — the FIRST ``_on_nav_selected`` for a
    module drops activation again as the freshly built screen is reparented
    into the stack. So this has to run after the navigation, not in the
    fixture. The ``waitUntil`` is the point: if activation ever stops
    landing, the test below fails here with "window never activated" rather
    than blaming the product for a focus call that worked.
    """
    window.raise_()
    window.activateWindow()
    qtbot.waitUntil(window.isActiveWindow, timeout=2000)


def test_ctrl_f_lands_in_the_settings_search_box_and_selects_the_old_query(
        window, qtbot):
    """Two claims, because ``_focus_settings_search`` makes two.

    The caret lands in the strip, and the previous query comes back
    SELECTED, so the next keystroke replaces the old search instead of
    appending to it — Ctrl+F twice in a row must not build up
    "diameterdiameter". Only the focus half was asserted before.
    """
    from spacr.qt.shortcuts import _focus_settings_search
    window._on_nav_selected("mask")
    qtbot.wait(50)
    screen = window._screens.get("mask")
    bar = getattr(screen, "_settings_search", None)
    assert bar is not None, "the strip was never installed"
    bar.set_query("merge")
    _activate(window, qtbot)

    _focus_settings_search(window)

    assert bar._input.hasFocus()
    assert bar._input.selectedText() == "merge"


def test_ctrl_f_is_harmless_on_a_screen_without_a_form(window):
    """"Harmless" is two claims, and neither is "it did not raise".

    Home has no settings strip, so ``_focus_settings_search`` has to leave
    the screen exactly as it found it — it returns early rather than
    swallowing the key or focusing something arbitrary.
    """
    from spacr.qt.shortcuts import _focus_settings_search
    window._on_nav_selected("__home__")
    home = window._stack.currentWidget()
    assert getattr(home, "_settings_search", None) is None, \
        "Home grew a settings strip; this test no longer covers the case"
    before = window.focusWidget()

    _focus_settings_search(window)

    assert window._stack.currentWidget() is home
    assert window.focusWidget() is before


# ---------------------------------------------------------------------------
# 2. The palette reaches settings
# ---------------------------------------------------------------------------

def _palette(window, qtbot):
    from spacr.qt.command_palette import CommandPalette
    palette = CommandPalette(window)
    qtbot.addWidget(palette)
    return palette


def test_the_palette_lists_the_current_modules_settings(window, qtbot):
    window._on_nav_selected("mask")
    qtbot.wait(50)
    palette = _palette(window, qtbot)
    settings = [c for c in palette._commands
                if c.section.startswith("Settings")]
    assert len(settings) == len(window._screens["mask"]._settings_model._widgets)


def test_a_setting_is_findable_in_the_palette_by_its_description(window,
                                                                  qtbot):
    """Same haystack as the settings strip: the description is the only part
    written in the language a user thinks in."""
    window._on_nav_selected("measure")
    qtbot.wait(50)
    palette = _palette(window, qtbot)
    palette._on_filter("straddling")
    labels = [palette._list.item(i).text()
              for i in range(palette._list.count())]
    assert any("merge_edge_pathogen_cells" in label for label in labels)


def test_activating_a_setting_reveals_it_on_the_form(window, qtbot):
    window._on_nav_selected("mask")
    qtbot.wait(50)
    screen = window._screens["mask"]
    palette = _palette(window, qtbot)
    palette._reveal_setting("merge_pathogens")

    bar = screen._settings_search
    assert bar.query() == "merge_pathogens"
    assert bar.visible_keys() == ["merge_pathogens"]
    holding = bar._index["merge_pathogens"][0]
    assert holding.is_expanded(), (
        "the palette named the setting and left it behind a closed heading")


def test_the_palette_scopes_settings_to_the_module_on_screen(window, qtbot):
    """Every setting of every module is over a thousand rows, and eleven
    identically-named `diameter` entries is a worse answer than none."""
    window._on_nav_selected("mask")
    qtbot.wait(50)
    sections = {c.section for c in _palette(window, qtbot)._commands}
    assert "Settings · mask" in sections
    assert "Settings · measure" not in sections


def test_the_palette_has_no_settings_section_on_home(window, qtbot):
    window._on_nav_selected("__home__")
    qtbot.wait(20)
    sections = {c.section for c in _palette(window, qtbot)._commands}
    assert not any(s.startswith("Settings") for s in sections)


def test_menu_commands_survive_being_collected(window, qtbot):
    """Menu actions are mirrored into the palette. Reaching them through
    ``QAction.menu()`` produced entries that raised "Internal C++ object
    already deleted" the moment they were triggered."""
    palette = _palette(window, qtbot)
    menu_cmds = [c for c in palette._commands if c.section == "Menu"]
    assert menu_cmds
    for cmd in menu_cmds:
        assert " → " in cmd.label
    labels = {c.label for c in menu_cmds}
    assert any("Settings recipes" in label for label in labels)


def test_installing_shortcuts_twice_is_harmless(window):
    install(window)  # already installed by MainWindow.__init__
    assert getattr(window, "_settings_search_watcher", None) is not None


def test_installing_them_twice_does_not_bind_anything_twice(window):
    """"Harmless" has to mean this, because for a shortcut it is not a
    nicety: two holders of one key make it AMBIGUOUS, and an ambiguous
    shortcut fires NEITHER handler. A reload path that called `install`
    again would have silenced every key it re-bound.
    """
    from PySide6.QtGui import QShortcut

    install(window)  # already installed by MainWindow.__init__

    keys = [sc.key().toString() for sc in window.findChildren(
        QShortcut, options=Qt.FindDirectChildrenOnly)]
    assert keys
    assert len(keys) == len(set(keys)), \
        [k for k in set(keys) if keys.count(k) > 1]
