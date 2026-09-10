"""Tests for keyboard shortcuts, OS notify, screen-reader labels."""
from __future__ import annotations


def test_shortcuts_spec_covers_every_binding():
    """Every ShortcutSpec must have keys + a label + a category."""
    from spacr.qt.shortcuts import SHORTCUTS
    assert len(SHORTCUTS) >= 10
    for s in SHORTCUTS:
        assert s.keys and s.label and s.category


def test_shortcuts_install_adds_qshortcuts(qtbot, qt_theme_applied):
    from PySide6.QtGui import QShortcut

    from spacr.qt import shortcuts
    from spacr.qt.app import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    # install() ran in MainWindow.__init__; count QShortcut children.
    scs = win.findChildren(QShortcut)
    # At least Ctrl+1..9 + Ctrl+K + Ctrl+/ + F1 + ?  (Ctrl+H and
    # Ctrl+P are menu actions, so install() creates no QShortcut.)
    #
    # `installed()`, NOT `SHORTCUTS`: backdrop, drawer and full-screen keys
    # are bound on window actions, so they are on the map and are not
    # install()'s to create.
    assert len(scs) >= len(shortcuts.installed())


def test_notify_signature_is_safe_when_no_backends(monkeypatch):
    """The notify function must never raise — silent no-op on error."""
    from spacr.qt import notify as n
    # Even if platform detection returns nothing usable, it returns False
    monkeypatch.setattr(n.platform, "system", lambda: "Plan9")
    assert n.notify("hello", "world") is False


def test_notify_esc_escapes_double_quotes():
    from spacr.qt.notify import _esc
    assert _esc('he "said"') == 'he \\"said\\"'
    assert _esc("") == ""
    assert _esc(None) == ""


def test_announce_pipeline_finished_falls_back_to_the_tray(monkeypatch):
    """When the OS backend declines, the tray gets the same message.

    Nothing about "it did not raise" is worth pinning: the reason this
    wrapper exists is the fallback, and the reason the fallback is useful
    is that it carries the module, the status and the elapsed time.
    """
    from spacr.qt import notify as n

    os_calls, tray_calls = [], []
    monkeypatch.setattr(n, "notify",
                        lambda *a, **k: (os_calls.append(a), False)[1])
    monkeypatch.setattr(n, "notify_tray",
                        lambda *a, **k: (tray_calls.append(a), False)[1])

    assert n.announce_pipeline_finished("mask", "success", 42.0) is None

    # The OS backend was tried first, with a message naming the module,
    # the status and the wall-clock time.
    assert len(os_calls) == 1
    title, body = os_calls[0][:2]
    assert "mask" in title and "success" in title and "✓" in title
    assert body == "Finished in 42.0s."
    # It said no, so the tray was asked with exactly the same message.
    assert tray_calls == [(title, body)]

    # Contrast: when the OS backend accepts, the tray is NOT double-fired.
    os_calls.clear()
    tray_calls.clear()
    monkeypatch.setattr(n, "notify",
                        lambda *a, **k: (os_calls.append(a), True)[1])
    n.announce_pipeline_finished("measure", "failed", 7.26)

    assert len(os_calls) == 1
    assert tray_calls == [], "the tray fired even though the OS backend took it"
    title, body = os_calls[0][:2]
    assert "measure" in title and "failed" in title and "⚠" in title
    assert body == "Finished in 7.3s."      # one decimal, rounded


def test_htile_has_accessibility_labels(qt_theme_applied):
    from spacr.qt.widgets.tile import HTile
    t = HTile(text="Mask", description="Segment cells.",
              icon=None, icon_size=32)
    assert t.accessibleName() == "Mask"
    assert t.accessibleDescription() == "Segment cells."


def test_sidebar_buttons_have_accessible_names(qtbot, qt_theme_applied):
    from PySide6.QtWidgets import QPushButton

    from spacr.qt.app import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    # At least the app buttons on the sidebar should carry accessible
    # names + descriptions.
    labeled = [
        b for b in win.findChildren(QPushButton)
        if b.accessibleName() and b.accessibleDescription()
    ]
    assert len(labeled) >= 5   # 5+ apps in APPS


def test_show_cheat_sheet_opens_and_closes(qtbot, qt_theme_applied):
    """The cheat sheet really appears, and lists every registered binding.

    It is an overlay over the window rather than a modal dialog now: ``?``
    asks a two-second question, and a modal makes the reader commit to a
    mode and find a close button to leave it. So the inspection is direct —
    no nested event loop, because nothing blocks — and closing it is the
    keystroke that a user would actually press.
    """
    from PySide6.QtWidgets import QLabel

    from spacr.qt.app import MainWindow
    from spacr.qt.shortcuts import (
        ShortcutOverlay,
        discover,
        mapped,
        native,
        show_cheat_sheet,
    )

    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1280, 860)

    overlay = show_cheat_sheet(win)
    qtbot.addWidget(overlay)
    assert isinstance(overlay, ShortcutOverlay), (
        "show_cheat_sheet never put an overlay on screen")
    assert overlay.parentWidget() is win

    rows = [lbl.text() for lbl in overlay._card.findChildren(QLabel)]
    # A title, a header per category, a keys+label pair per binding, and the
    # closing hint. All non-empty.
    #
    # `mapped()`, NOT `SHORTCUTS` (197). The map describes the per-screen
    # keys too -- the Make Masks tools, the Annotate navigation -- and they
    # are declared in `SCREEN_SHORTCUTS` because a screen binds them and the
    # map has to name them whether that screen is built or not.
    # The card also includes window actions discovered at runtime (for
    # example the platform's Quit action), exactly as the implementation
    # promises. Count and inspect the same declared-plus-live inventory.
    specs = mapped() + discover(win)
    categories = {s.category for s in specs}
    assert len(rows) == 2 * len(specs) + len(categories) + 2
    assert all(r.strip() for r in rows)

    text = "\n".join(rows)
    for spec in specs:
        # PRINTED IN THE PLATFORM'S SPELLING: `Ctrl` is the Command symbol
        # on macOS, so the card is searched for what it actually shows.
        assert native(spec.keys) in text, \
            f"{spec.keys} missing from the cheat sheet"
        assert spec.label in text, f"{spec.label!r} missing from the cheat sheet"
    for cat in categories:
        assert cat.upper() in text

    # Contrast: the search above is discriminating, not "in a big blob of
    # text everything matches" — a binding that is not registered is absent.
    assert "Ctrl+Shift+Q" not in text
    assert "Reticulate splines" not in text

    overlay.dismiss()
    assert not overlay.isVisible()


def test_no_key_is_held_by_both_a_menu_action_and_a_qshortcut(
        qtbot, qt_theme_applied):
    """A key with two holders fires NEITHER of them.

    Reported 2026-09-08: Ctrl+H and Ctrl+, both logged "QAction::event:
    Ambiguous shortcut overload" and did nothing. The menu bar builds
    Home and Preferences as QActions carrying those sequences so the menu
    can print the accelerator beside the item, and `shortcuts.install`
    bound a QShortcut for the same two keys on the same window. Qt
    resolves that by refusing to guess. Preferences moved to Ctrl+P in
    the same report; Home kept Ctrl+H, which works once it has one
    holder.

    `_bind` had a de-duplication loop already, and it was blind here: it
    searched the window's QShortcut children, and a QAction is not one.
    The bug was not a missing guard but a guard that looked in one of the
    two places a shortcut can live.

    Asserted over the WHOLE window rather than the two known keys,
    because the next menu item to be given an accelerator will collide
    the same way and there is nothing about those two keys in particular.
    """
    from PySide6.QtGui import QShortcut

    from spacr.qt.app import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)

    action_keys = {}
    for action in win.findChildren(type(win.actions()[0]) if win.actions()
                                   else object):
        sequence = action.shortcut()
        if not sequence.isEmpty():
            action_keys.setdefault(sequence.toString(), []).append(
                action.text())

    clashes = {}
    for shortcut in win.findChildren(QShortcut):
        name = shortcut.key().toString()
        if name in action_keys:
            clashes[name] = action_keys[name]

    assert not clashes, (
        "these keys are held by a menu QAction AND a QShortcut, so Qt "
        f"fires neither: {clashes}"
    )


def test_the_home_and_preferences_keys_actually_reach_their_action(
        qtbot, qt_theme_applied):
    """Not ambiguous is not the same as working.

    Removing one of two holders fixes the "fires neither" symptom and can
    equally leave a key that reaches nothing at all, so the keys are
    PRESSED here rather than reasoned about.

    Two things swallow a press in this environment and BOTH look exactly
    like a dead shortcut, which is most of why this test exists: the
    first-run tour overlay, and the first press after ``show`` while the
    window is still becoming active. Ctrl+H was briefly believed unusable
    on Linux on the strength of a check that had not accounted for the
    second one.

    Each action's own slot is replaced with a recorder first: the real
    Preferences slot opens a modal dialog, and a test that opens one
    never returns.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QAction
    from PySide6.QtTest import QTest

    from spacr.qt.app import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)

    # THE FIRST-RUN TOUR TAKES FOCUS AND KEEPS THE KEYS. The store is
    # isolated per test, so every window built here is a first run and the
    # overlay is up; a press aimed at the window reaches the overlay
    # instead of the shortcut map, and Ctrl+H read as dead when it was
    # only covered. Escape is the tour's own skip, so this dismisses it
    # the way a user would rather than reaching into its internals.
    from spacr.qt.first_run import _TourOverlay
    for overlay in win.findChildren(_TourOverlay):
        QTest.keyClick(overlay, Qt.Key_Escape)
        qtbot.wait(10)
    assert not [o for o in win.findChildren(_TourOverlay) if o.isVisible()], (
        "the first-run tour is still up; it would swallow the presses below")

    # AND THE FIRST PRESS AFTER `show` IS LOST while the window becomes
    # active, which reads exactly like a dead key: whichever key is
    # pressed first fails and the rest pass. Spend that press on a
    # sequence nothing claims, so the real ones are all sent to a settled
    # window.
    win.activateWindow()
    QTest.keyClick(win, Qt.Key_F12, Qt.ControlModifier | Qt.AltModifier)
    qtbot.wait(20)

    fired = []
    wanted = {"Ctrl+H": Qt.Key_H, "Ctrl+P": Qt.Key_P}
    seen = {}
    for action in win.findChildren(QAction):
        name = action.shortcut().toString()
        if name in wanted:
            seen[name] = action
            action.triggered.disconnect()
            action.triggered.connect(
                lambda _checked=False, key=name: fired.append(key))

    assert set(seen) == set(wanted), (
        f"no menu action carries these keys: {set(wanted) - set(seen)}")

    for name, key in wanted.items():
        QTest.keyClick(win, key, Qt.ControlModifier)
        qtbot.wait(10)

    assert sorted(fired) == sorted(wanted), (
        f"pressed {sorted(wanted)}, reached {sorted(fired)}")
