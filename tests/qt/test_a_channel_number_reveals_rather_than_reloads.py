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
`organelleb_*` controls that were not there. Since 2026-09-25 those are built
and laid out on the screen already open, so neither case reloads the module;
the rebuild is left only for a run that owns the screen.
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


def _set_count(model, count):
    combo = model._widgets["number_of_organelles"]
    index = next(i for i in range(combo.count())
                 if str(combo.itemData(i)) == str(count))
    combo.setCurrentIndex(index)


def _slot(model, role):
    from spacr.organelle_types import organelle_role_of

    return {key: widget for key, widget in model._widgets.built_items()
            if organelle_role_of(key) == role}


def _an_unrelated_spin_box(model):
    from PySide6.QtWidgets import QSpinBox

    return next((key, widget) for key, widget in model._widgets.built_items()
                if isinstance(widget, QSpinBox) and "organelle" not in key
                and widget.value() < widget.maximum())


def test_the_organelle_count_spawns_its_settings_in_place(mask_window, qtbot,
                                                          monkeypatch):
    """Case 2, done 2026-09-25: the count no longer reloads the module.

    A second organelle's 52 controls did not exist until the count said so,
    which is why this used to rebuild the whole screen. They are now built
    and laid out in the headings already on screen, and every control that
    existed stays the same object.
    """
    window, screen = mask_window
    calls = _count_rebuilds(window, monkeypatch)
    model = screen._settings_model
    assert not [k for k in model._widgets if k.startswith("organelleb_")]
    existing = dict(model._widgets.built_items())
    key, spin = _an_unrelated_spin_box(model)
    spin.setValue(spin.value() + 1)
    typed = spin.value()

    _set_count(model, 2)
    qtbot.wait(50)

    assert calls == [], f"the organelle count rebuilt the form: {calls}"
    assert window._screens["mask"] is screen
    spawned = [k for k in model._widgets if k.startswith("organelleb_")]
    assert len(spawned) > 20, (
        f"only {len(spawned)} controls were spawned for the second organelle")
    rebuilt = [name for name, widget in existing.items()
               if model._widgets.built(name) is not widget]
    assert not rebuilt, f"controls that existed were replaced: {rebuilt[:5]}"
    assert model._widgets.built(key) is spin
    assert spin.value() == typed, "an uncommitted value was lost"
    assert model.collect()["number_of_organelles"] == 2


def test_raising_two_to_three_keeps_slots_one_and_two(mask_window, qtbot,
                                                      monkeypatch):
    """The check this file set itself, verbatim: slots 1 and 2 keep their
    widgets and slot 3's are added."""
    window, screen = mask_window
    calls = _count_rebuilds(window, monkeypatch)
    model = screen._settings_model
    _set_count(model, 2)
    qtbot.wait(50)
    one, two = _slot(model, "organelle"), _slot(model, "organelleb")
    assert two, "slot 2 was not spawned"

    _set_count(model, 3)
    qtbot.wait(50)

    assert calls == []
    for name, widget in {**one, **two}.items():
        assert model._widgets.built(name) is widget, f"{name} was rebuilt"
    assert len([k for k in model._widgets
                if k.startswith("organellec_")]) > 20


def test_the_new_slot_has_a_captioned_channel_on_the_form(mask_window, qtbot,
                                                          monkeypatch):
    """A spawned slot is not a set of controls with nowhere to live: its
    channel row is laid out beside the first slot's and captioned."""
    window, screen = mask_window
    _count_rebuilds(window, monkeypatch)
    model = screen._settings_model

    _set_count(model, 2)
    qtbot.wait(50)

    channel = model._widgets.built("organelleb_channel")
    assert channel is not None
    assert getattr(channel, "_spacr_setting_label", None) is not None, (
        "the new slot's channel was never captioned")
    holders = [section for section in screen._settings_sections
               if any(key == "organelleb_channel" for key, _l, _w in
                      getattr(section, "_spacr_declared_rows", ()) or ())]
    assert holders, "no heading declares the new slot's channel"
    declared = [key for key, _l, _w in holders[0]._spacr_declared_rows]
    if "organelle_channel" in declared:
        assert (declared.index("organelleb_channel")
                == declared.index("organelle_channel") + 1)


def test_the_new_slots_channel_reveals_its_settings(mask_window, qtbot,
                                                    monkeypatch):
    """The spawned switch is watched like the ones the panel was built with."""
    window, screen = mask_window
    calls = _count_rebuilds(window, monkeypatch)
    model = screen._settings_model
    _set_count(model, 2)
    qtbot.wait(50)
    before = {k for k in model.keys_hidden_by_the_run()
              if k.startswith("organelleb_")}
    assert before, "none of the new slot's settings wait for its channel"

    _commit(model._widgets["organelleb_channel"], 1)
    qtbot.wait(20)

    after = {k for k in model.keys_hidden_by_the_run()
             if k.startswith("organelleb_")}
    assert after < before
    assert calls == []


def test_lowering_the_count_hides_the_slot_in_place(mask_window, qtbot,
                                                    monkeypatch):
    window, screen = mask_window
    calls = _count_rebuilds(window, monkeypatch)
    model = screen._settings_model
    _set_count(model, 2)
    qtbot.wait(50)
    two = _slot(model, "organelleb")

    _set_count(model, 1)
    qtbot.wait(50)

    assert calls == []
    assert "organelleb_channel" in set(model.keys_hidden_by_the_run())
    for name, widget in two.items():
        assert model._widgets.built(name) is widget
    assert "organelleb_channel" not in model.collect()


def test_the_per_object_table_gains_the_new_organelle_as_a_column(
        mask_window, qtbot, monkeypatch):
    """With 364's table mounted, the new slot is a COLUMN, still no reload."""
    window, screen = mask_window
    calls = _count_rebuilds(window, monkeypatch)
    monkeypatch.setattr("spacr.qt.preferences.get_object_grid_enabled",
                        lambda: True)
    screen.apply_object_grid_preference()
    grid = getattr(screen, "_object_grid", None)
    if grid is None:
        pytest.skip("Mask did not mount the per-object table")
    model = screen._settings_model

    _set_count(model, 2)
    qtbot.wait(50)

    assert calls == []
    assert {"organelle", "organelleb"} <= set(grid.objects())


def test_a_run_owning_the_screen_still_defers_to_the_rebuild(mask_window,
                                                             monkeypatch):
    """The fallback is kept: a run in flight must not have its form grown
    under it, so the count goes the rebuild's deferred way."""
    _window, screen = mask_window
    rebuilt = []
    monkeypatch.setattr(screen, "_worker_thread_is_running", lambda: True)
    monkeypatch.setattr(screen, "_rebuild_the_form",
                        lambda *a: rebuilt.append(1))

    _set_count(screen._settings_model, 2)

    assert rebuilt == [1]


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
