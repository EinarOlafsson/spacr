"""Typing a channel number toggles a category; it does not reload the module.

Instruction 356, in the maintainer's words: "when a number is added to any of
the object channel settings, or number of organels, the entire module reloads
. i would like the presense of integers in these settings to toggle visability
of their corresponding settings categories without reloading the entire
module".

MEASURED ON MASK, before any of this was changed: committing a value into
`nucleus_channel` called `rebuild_app_screen`, took 455 ms, and put a
DIFFERENT SCREEN OBJECT in the window's stack. Not one row was added or
removed by it -- the panel has already built a control for every object it
can name -- so the entire cost bought a change of visibility. Afterwards: no
rebuild, 18 ms, same screen.

THE TWO CASES ARE NOT THE SAME and the request separates them itself. A
channel only decides which of the rows already on the form are SHOWN.
`number_of_organelles` decides which rows EXIST -- raising it to 2 spawns 52
`organelleb_*` controls that were not there -- and the same request accepts
that those "have to be spawned". So the rebuild is kept for exactly that key
and pinned here, because silently losing it would be a slot with no settings.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def mask_window(qtbot, qt_theme_applied):
    """A real window with Mask open, which is the only place this is wired.

    The watcher is installed as the screen is built by the WINDOW; a screen
    constructed on its own has no `rebuild_app_screen` to reach and would let
    a rebuild-on-keystroke pass unnoticed.
    """
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    window.open_module("mask")
    qtbot.wait(20)
    screen = window._screens.get("mask")
    assert screen is not None, "Mask did not open"
    return window, screen


def _count_rebuilds(window, monkeypatch):
    """Record every `rebuild_app_screen` call without preventing it."""
    calls = []
    real = window.rebuild_app_screen

    def counting(key, keep=None):
        calls.append(key)
        return real(key, keep)

    monkeypatch.setattr(window, "rebuild_app_screen", counting)
    return calls


def _commit(widget, value):
    """Set a value and commit it the way leaving the field does."""
    widget.set_value(value)
    widget.editingFinished.emit()


def test_a_channel_does_not_reload_the_module(mask_window, qtbot, monkeypatch):
    """The request itself: no reload for a number in a channel box."""
    window, screen = mask_window
    calls = _count_rebuilds(window, monkeypatch)

    _commit(screen._settings_model._widgets["nucleus_channel"], 1)
    qtbot.wait(20)

    assert calls == [], f"typing a channel number rebuilt the form: {calls}"
    assert window._screens.get("mask") is screen, (
        "the screen object was replaced, so anything uncommitted is gone")


def test_the_category_appears_anyway(mask_window, qtbot, monkeypatch):
    """Not reloading must not mean not reacting: the rows the object owns
    have to come back, which is what the reload was reaching for."""
    window, screen = mask_window
    _count_rebuilds(window, monkeypatch)
    model = screen._settings_model
    before = set(model.keys_hidden_by_the_run())

    _commit(model._widgets["nucleus_channel"], 1)
    qtbot.wait(20)

    after = set(model.keys_hidden_by_the_run())
    assert after != before, "the object rule did not re-run at all"
    assert len(after) < len(before), (
        f"giving nucleus a channel hid MORE than before: {before} -> {after}")


def test_clearing_it_hides_them_again(mask_window, qtbot, monkeypatch):
    """Idempotent in both directions, which a one-way reveal would not be."""
    window, screen = mask_window
    _count_rebuilds(window, monkeypatch)
    model = screen._settings_model
    start = set(model.keys_hidden_by_the_run())

    _commit(model._widgets["nucleus_channel"], 1)
    qtbot.wait(20)
    _commit(model._widgets["nucleus_channel"], None)
    qtbot.wait(20)

    assert set(model.keys_hidden_by_the_run()) == start


def test_the_organelle_count_still_spawns_its_settings(mask_window, qtbot,
                                                       monkeypatch):
    """The deliberate exception, pinned so it cannot be optimised away.

    A second organelle's settings do not exist until the count says so:
    measured, raising it to 2 creates 52 `organelleb_*` controls where there
    were none. That is a rebuild, and the request accepts it.
    """
    window, screen = mask_window
    calls = _count_rebuilds(window, monkeypatch)
    model = screen._settings_model
    combo = model._widgets["number_of_organelles"]
    assert not [k for k in model._widgets if k.startswith("organelleb_")]

    index = next(i for i in range(combo.count())
                 if str(combo.itemData(i)) == "2")
    combo.setCurrentIndex(index)
    qtbot.wait(50)

    assert calls == ["mask"], f"the organelle count did not rebuild: {calls}"
    fresh = window._screens["mask"]
    spawned = [k for k in fresh._settings_model._widgets
               if k.startswith("organelleb_")]
    assert len(spawned) > 20, (
        f"only {len(spawned)} controls were spawned for the second organelle")


def test_the_two_kinds_of_key_are_kept_apart(mask_window):
    """The split itself, so a key added to one list is not quietly in both."""
    _window, screen = mask_window

    switches = set(screen._object_switches_on_this_form())
    shaping = set(screen._form_shaping_keys())

    assert "number_of_organelles" not in switches
    assert "number_of_organelles" in shaping
    assert switches, "no object switch was found on Mask's form"
    assert switches < shaping, "a switch that no longer shapes the form"
    assert not [k for k in switches if k in screen.FORM_SHAPING_KEYS]
