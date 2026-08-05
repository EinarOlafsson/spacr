"""Per-module walkthroughs, and the menu lookup that used to segfault.

Two things are pinned here. The first is that a walkthrough is *per module*
and *re-runnable* — one global "seen" flag set on the day the app was first
opened is exactly what made the old tour useless to everybody afterwards.
The second is that menus are found through ``findChildren``, because the
obvious reading of the same lookup does not survive on PySide6 6.11.
"""
from __future__ import annotations

import pytest

from PySide6.QtWidgets import QMainWindow, QMenu

from spacr.qt import walkthrough as W


@pytest.fixture(autouse=True)
def _clean_seen():
    W.reset()
    yield
    W.reset()


@pytest.fixture
def window(qtbot):
    from spacr.qt.app import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1200, 800)
    # Installed explicitly rather than relied on from
    # `shortcuts._install_window_hooks`, so this file tests the walkthrough
    # rather than the wiring — and passes at the commit that adds it.
    W.install_window_hooks(win)
    return win


# ---------------------------------------------------------------------------
# 1. Per module, and re-runnable
# ---------------------------------------------------------------------------

def test_seen_is_tracked_per_module():
    """A module added next release has to introduce itself to a user who
    has been running spaCR for a year, which one global flag cannot do."""
    assert not W.was_seen("mask")
    W.mark_seen("mask")
    assert W.was_seen("mask")
    assert not W.was_seen("measure")


def test_a_walkthrough_can_be_run_again_after_it_was_seen(window, qtbot):
    first = W.show_walkthrough(window, "mask")
    assert first is not None
    qtbot.addWidget(first)
    first._skip()
    assert W.was_seen("mask")

    again = W.show_walkthrough(window, "mask")
    assert again is not None, (
        "a walkthrough that can only ever run once is not re-runnable")
    qtbot.addWidget(again)
    again._skip()


def test_the_automatic_route_respects_seen(window, qtbot):
    """Automatic on first open, silent afterwards — the only place the flag
    is allowed to stop anything."""
    W.mark_seen("mask")
    assert W.maybe_show(window, "mask") is None
    W.reset("mask")
    overlay = W.maybe_show(window, "mask")
    assert overlay is not None
    qtbot.addWidget(overlay)
    overlay._skip()


def test_finishing_marks_only_that_module(window, qtbot):
    overlay = W.show_walkthrough(window, "measure")
    qtbot.addWidget(overlay)
    for _ in range(len(overlay._steps) + 1):
        overlay._next()
    assert W.was_seen("measure")
    assert not W.was_seen("mask")


def test_reset_brings_every_walkthrough_back():
    W.mark_seen("mask")
    W.mark_seen("measure")
    W.reset()
    assert not W.was_seen("mask")
    assert not W.was_seen("measure")


# ---------------------------------------------------------------------------
# 2. Steps are derived, so they cannot go stale
# ---------------------------------------------------------------------------

def test_steps_name_the_modules_own_first_group():
    """Hand-written prose about a settings panel rots the first time the
    panel is regrouped. These sentences are read out of the layout."""
    from spacr.qt.screens.settings_model import (
        categories_for_app, get_categories,
    )
    first = list(categories_for_app("mask", get_categories()))[0]
    bodies = " ".join(s.body for s in W.build_steps("mask"))
    assert first in bodies


def test_steps_state_the_real_essentials_and_the_real_total():
    from spacr.qt.screens.settings_model import (
        essential_keys, resolve_default_settings,
    )
    total = len(resolve_default_settings("mask"))
    essentials = len([k for k in essential_keys("mask")
                      if k in resolve_default_settings("mask")])
    titles = " ".join(s.title for s in W.build_steps("mask"))
    assert str(total) in titles
    assert str(essentials) in titles


def test_every_module_produces_a_usable_walkthrough():
    from spacr.qt.app import APPS
    for key, _name, _desc, _section in APPS:
        steps = W.build_steps(key)
        assert steps, f"{key} has no walkthrough at all"
        assert all(s.title and s.body for s in steps), key


def test_a_module_can_register_advice_the_layout_cannot_express():
    """"Run test mode on three fields first" is advice, not structure; no
    amount of reading the settings map produces it."""
    W.unregister_steps("mask")
    W.register_steps("mask", [W.WalkStep("Test first", "Three fields.")])
    try:
        assert W.build_steps("mask")[-1].title == "Test first"
        with pytest.raises(ValueError, match="already registered"):
            W.register_steps("mask", [])
        W.register_steps("mask", [], replace=True)
    finally:
        W.unregister_steps("mask")


# ---------------------------------------------------------------------------
# 3. The menu
# ---------------------------------------------------------------------------

def test_the_help_menu_lists_every_visible_module(window):
    submenu = window._walkthrough_menu
    assert isinstance(submenu, QMenu)
    labels = [a.text() for a in submenu.actions() if not a.isSeparator()]
    from spacr.qt.app import APPS, app_is_visible
    for _key, name, _desc, _section in APPS:
        if app_is_visible(_key):
            assert name in labels, f"{name} has no walkthrough entry"
    assert "Show all walkthroughs again" in labels


def test_the_submenu_is_installed_once(window):
    assert W.install_help_menu(window) is None
    from spacr.qt.first_run import find_menu
    help_menu = find_menu(window, "Help")
    titles = [a.text().replace("&", "") for a in help_menu.actions()]
    assert titles.count(W.MENU_TITLE) == 1


def test_a_menu_entry_runs_its_own_module(window, qtbot, monkeypatch):
    shown = []
    monkeypatch.setattr(
        W, "show_walkthrough",
        lambda win, key, **kw: shown.append(key))
    submenu = window._walkthrough_menu
    for action in submenu.actions():
        if action.text() == "Measure":
            action.trigger()
            break
    assert shown == ["measure"]


def test_the_reset_entry_clears_every_seen_flag(window):
    W.mark_seen("mask")
    for action in window._walkthrough_menu.actions():
        if action.text() == "Show all walkthroughs again":
            action.trigger()
    assert not W.was_seen("mask")


# ---------------------------------------------------------------------------
# 4. The menu lookup that used to die
# ---------------------------------------------------------------------------
#
# `QAction.menu()` hands back a QMenu wrapper that is only valid while the
# QAction wrapper it came off is alive. Walking `menuBar().actions()` and
# calling it therefore produced a menu that was already dead by the time the
# calling function returned — "Internal C++ object already deleted" — and
# keeping the owners alive as attributes segfaulted during the next event
# dispatch instead. `findChildren` hands back children the bar owns in C++.

def test_find_menu_survives_the_call_that_produced_it(window):
    from spacr.qt.first_run import find_menu
    menu = find_menu(window, "Help")
    assert menu is not None
    # The lookup's locals are gone by now. Touching the menu is what used to
    # raise; the assertion is that it does not.
    assert menu.title().replace("&", "") == "Help"
    assert menu.actions()


def test_find_menu_answers_none_for_a_bare_window(qtbot):
    from spacr.qt.first_run import find_menu
    bare = QMainWindow()
    qtbot.addWidget(bare)
    assert find_menu(bare, "Nothing") is None


#: The one remaining site, and why it is still here.
#:
#: ``AppScreen._open_demos_menu`` pops the Demos menu on click. It has the
#: same lifetime hazard as the two fixed here — it walks the bar's action
#: list and calls ``QAction.menu()`` — but it is inside a file this change
#: does not own, and it survives in practice only because it uses the menu
#: within the same statement. Named rather than silently skipped so it is a
#: known debt with a test attached rather than a rediscovery.
_KNOWN_MENU_WALKERS = {"app_screen.py"}


def _menu_bar_action_walkers():
    """Every ``for x in <...>.menuBar().actions()`` in the Qt package.

    Found by parsing rather than by grepping, so the prose in a docstring
    explaining why the pattern is banned does not itself count as an
    instance of it.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "spacr" / "qt"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken file is its own bug
            continue
        for node in ast.walk(tree):
            call = getattr(node, "iter", None)
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if not (isinstance(func, ast.Attribute) and func.attr == "actions"):
                continue
            inner = func.value
            if isinstance(inner, ast.Call) and isinstance(
                    inner.func, ast.Attribute) and \
                    inner.func.attr == "menuBar":
                offenders.append(f"{path.name}:{node.lineno}")
    return offenders


def test_no_qt_module_reaches_a_menu_through_the_action_walk():
    """The fix is that the pattern is gone, not that one call site changed.

    ``QAction.menu()`` is still legal where the action IS the submenu's
    owner. What is banned is using it to FIND a menu bar's menu — walking
    ``menuBar().actions()`` — because the QMenu it hands back dies with the
    QAction wrapper it came off.
    """
    offenders = [o for o in _menu_bar_action_walkers()
                 if o.split(":")[0] not in _KNOWN_MENU_WALKERS]
    assert not offenders, (
        "these walk the menu bar's action list to find a menu; use "
        f"`bar.findChildren(QMenu)` instead: {offenders}")


def test_the_known_menu_walker_is_still_the_only_one():
    """Fails when the last site is fixed, so the exemption is removed with
    it rather than outliving it."""
    remaining = {o.split(":")[0] for o in _menu_bar_action_walkers()}
    assert remaining == _KNOWN_MENU_WALKERS, (
        f"update _KNOWN_MENU_WALKERS: it now reads {sorted(remaining)}")


def test_the_tutorial_menu_target_uses_the_safe_lookup(window):
    from spacr.qt.tutorial.scripts import _find_menu, _menu_target
    assert _find_menu(window, "Demos") is not None
    bar, point = _menu_target(window, "Demos")
    assert bar is window.menuBar()
    assert point is not None and point[0] > 0


def test_the_tutorial_demos_step_resolves_a_live_menu(window):
    from spacr.qt.tutorial.scripts import _open_demos_menu
    menu = _open_demos_menu(window)
    assert menu is not None
    assert menu.title().replace("&", "") == "Demos"
