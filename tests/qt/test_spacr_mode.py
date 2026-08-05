"""spaCR mode: when a cleanup runs, and — mostly — when it does not.

Three settings, and the only interesting claim is negative. Balanced is the
default and Balanced must *never* clean up: not at launch, not before a run,
not quietly on the way past. A mode that hedges — "mostly nothing, but it
does collect once at startup" — is a mode nobody can reason about, and the
user asked for the visuals to stay exactly as they were set.

So every test here asserts on a recorded list of calls rather than on an
outcome, because "it freed nothing" and "it did not run" look the same from
the outside and only one of them is what Balanced promises.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings

from spacr.qt import preferences as prefs
from spacr.qt import resource_cleanup as rc


@pytest.fixture(autouse=True)
def _isolated_qsettings(monkeypatch, qt_theme_applied, tmp_path):
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    return store


@pytest.fixture
def calls(monkeypatch):
    """Record every cleanup this module can perform, and perform none."""
    recorded = []
    monkeypatch.setattr(rc, "clear_ram",
                        lambda **kw: (recorded.append(("ram", kw)),
                                      rc.Reclaim("ram"))[1])
    monkeypatch.setattr(rc, "clear_vram",
                        lambda **kw: (recorded.append(("vram", kw)),
                                      rc.Reclaim("vram"))[1])
    monkeypatch.setattr(rc, "clear_cpu",
                        lambda **kw: (recorded.append(("cpu", kw)),
                                      rc.Reclaim("cpu"))[1])
    return recorded


def _actions(recorded):
    return [name for name, _kw in recorded]


# ---------------------------------------------------------------------------
# The setting
# ---------------------------------------------------------------------------

def test_balanced_is_the_default():
    assert prefs.DEFAULT_SPACR_MODE == "balanced"
    assert prefs.get_spacr_mode() == "balanced"
    assert set(prefs.SPACR_MODES) == {"extra_performance", "performance",
                                      "balanced"}


def test_each_mode_round_trips():
    for mode in prefs.SPACR_MODES:
        prefs.set_spacr_mode(mode)
        assert prefs.get_spacr_mode() == mode


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        prefs.set_spacr_mode("turbo")
    assert prefs.get_spacr_mode() == "balanced"


def test_a_corrupt_stored_mode_falls_back_to_balanced(_isolated_qsettings):
    _isolated_qsettings.setValue("prefs/spacr_mode", "ludicrous")
    assert prefs.get_spacr_mode() == "balanced"


def test_both_performance_modes_warn_and_balanced_does_not():
    for mode in ("extra_performance", "performance"):
        warning = prefs.mode_warning(mode)
        assert warning, f"{mode} costs something and must say so"
        assert len(warning.split()) >= 20
        assert "never touches another program" in warning
    assert prefs.mode_warning("balanced") == ""


def test_every_mode_explains_when_it_cleans_up():
    assert "before every module run" in prefs.mode_note("extra_performance")
    assert "at launch" in prefs.mode_note("performance")
    note = prefs.mode_note("balanced")
    assert "Nothing is freed at launch or before a run" in note
    assert "exactly as you set them" in note


# ---------------------------------------------------------------------------
# When a cleanup runs
# ---------------------------------------------------------------------------

def test_balanced_never_cleans_up_at_launch(calls):
    prefs.set_spacr_mode("balanced")
    assert rc.run_launch_cleanup() == []
    assert calls == [], f"Balanced ran {_actions(calls)} at launch"


def test_balanced_never_cleans_up_before_a_run(calls):
    prefs.set_spacr_mode("balanced")
    assert rc.run_pre_run_cleanup("measure") == []
    assert calls == []


def test_performance_cleans_up_at_launch_only(calls):
    prefs.set_spacr_mode("performance")
    rc.run_launch_cleanup()
    assert _actions(calls) == ["ram", "vram"]
    calls.clear()
    rc.run_pre_run_cleanup("measure")
    assert calls == [], "Performance cleaned up before a run; it must not"


def test_performance_still_cleans_up_when_a_button_is_pressed(monkeypatch):
    """"At launch only" is about the automatic cleanups, not the buttons."""
    ran = []
    monkeypatch.setattr(prefs, "confirm_resource_action", lambda *a, **k: True)
    monkeypatch.setattr(rc, "clear_ram",
                        lambda **kw: (ran.append("ram"), rc.Reclaim("ram"))[1])
    monkeypatch.setattr(prefs, "_show_resource_result", lambda *a, **k: None)
    prefs.set_spacr_mode("performance")
    prefs.run_resource_action("ram")
    assert ran == ["ram"]


def test_extra_performance_cleans_up_at_launch_and_before_every_run(calls):
    prefs.set_spacr_mode("extra_performance")
    rc.run_launch_cleanup()
    assert _actions(calls) == ["ram", "vram", "cpu"]
    calls.clear()
    rc.run_pre_run_cleanup("measure")
    assert _actions(calls) == ["ram", "vram"]
    calls.clear()
    rc.run_pre_run_cleanup("mask")
    assert _actions(calls) == ["ram", "vram"]


def test_the_pre_run_cleanup_does_not_fight_the_run_it_precedes(calls):
    """Three separate ways it stays out of the way, all asserted.

    Releasing a model the run is about to reload is a slowdown dressed as
    an optimisation, so the pre-run pass keeps the model references; and it
    does not touch the CPU, because lowering the thread count a moment
    before a run that wants those threads is the same mistake in another
    resource.
    """
    prefs.set_spacr_mode("extra_performance")
    rc.run_pre_run_cleanup("measure")
    by_action = dict(calls)
    assert by_action["vram"]["release_models"] is False
    assert "cpu" not in by_action
    # ...and the launch pass, where nothing is running, does release them.
    calls.clear()
    rc.run_launch_cleanup()
    assert dict(calls)["vram"].get("release_models", True) is True


def test_the_pre_run_cleanup_stands_down_while_another_run_is_going(
        calls, monkeypatch):
    """The caches it drops are the ones a running job is reading."""
    class _Registry:
        def active(self):
            return [object(), object()]

    import spacr.qt.bridge as bridge
    monkeypatch.setattr(bridge, "registry", lambda: _Registry())
    prefs.set_spacr_mode("extra_performance")
    assert rc.run_pre_run_cleanup("measure") == []
    assert calls == []


def test_the_launch_cleanup_happens_once_per_process(calls, monkeypatch):
    monkeypatch.setattr(rc, "_LAUNCH_DONE", False)
    monkeypatch.setattr(rc, "_INSTALLED", False)
    prefs.set_spacr_mode("performance")
    assert rc.register() is True
    first = list(_actions(calls))
    assert first, "the launch cleanup did not run at all"
    assert rc.register() is True
    assert _actions(calls) == first, "it cleaned up twice in one process"


# ---------------------------------------------------------------------------
# The hook that makes "before every module run" true
# ---------------------------------------------------------------------------

def test_a_new_run_triggers_the_pre_run_cleanup(calls, monkeypatch):
    """Driven through the real registry, not through a call to the hook.

    ``RunRegistry.changed`` is emitted from inside ``register()``, which
    ``make_thread`` calls before it hands the *unstarted* thread back — so
    the cleanup really does happen before the worker runs, without a line
    of this feature living inside bridge.py.
    """
    from spacr.qt import bridge

    monkeypatch.setattr(rc, "_INSTALLED", False)
    monkeypatch.setattr(rc, "_SEEN_RUNS", set())
    prefs.set_spacr_mode("extra_performance")
    assert rc.install_run_hook() is True

    registry = bridge.registry()
    thread = worker = None
    try:
        thread, worker = bridge.make_thread(lambda settings: None, {},
                                            app_key="measure", journal=False)
        assert _actions(calls) == ["ram", "vram"], (
            "starting a job did not trigger the pre-run cleanup")
        calls.clear()
        # The same registry event firing again (another job finishing, a
        # progress update) must not clean up a second time.
        registry.changed.emit()
        assert calls == []
    finally:
        registry.clear()
        del thread, worker


def test_the_hook_is_installed_at_launch():
    from spacr.qt import SELF_REGISTERING_MODULES
    assert "spacr.qt.resource_cleanup" in SELF_REGISTERING_MODULES
    index = SELF_REGISTERING_MODULES.index("spacr.qt.resource_cleanup")
    assert SELF_REGISTERING_MODULES[-1] == "spacr.qt.maturity", (
        "maturity has to stay last; this row goes before it")
    assert index < len(SELF_REGISTERING_MODULES) - 1


def test_installing_the_hook_twice_connects_it_once(monkeypatch):
    monkeypatch.setattr(rc, "_INSTALLED", False)
    assert rc.install_run_hook() is True
    assert rc.install_run_hook() is True


# ---------------------------------------------------------------------------
# What each mode does to the visual settings
# ---------------------------------------------------------------------------

def test_balanced_leaves_every_visual_setting_alone():
    prefs.set_ambient_animation("cells")
    prefs.set_setting_animations_enabled(True)
    prefs.set_field_fade_enabled(True)
    before = prefs._visual_snapshot()
    prefs.set_spacr_mode("balanced")
    assert prefs._visual_snapshot() == before


def test_performance_leaves_every_visual_setting_alone():
    prefs.set_ambient_animation("aurora")
    prefs.set_setting_animations_enabled(True)
    prefs.set_field_fade_enabled(True)
    before = prefs._visual_snapshot()
    prefs.set_spacr_mode("performance")
    assert prefs._visual_snapshot() == before


def test_extra_performance_drops_every_visual_setting_to_its_minimum():
    from spacr.qt.widgets import ambient

    prefs.set_ambient_animation("cells")
    prefs.set_ambient_resolution(2.0)
    prefs.set_ambient_density(2.0)
    prefs.set_setting_animations_enabled(True)
    prefs.set_field_fade_enabled(True)

    prefs.set_spacr_mode("extra_performance")

    assert prefs.get_ambient_animation() == ambient.NO_ANIMATION
    assert prefs.get_ambient_enabled() is False
    assert prefs.get_ambient_resolution() == ambient.RESOLUTION_RANGE[0]
    assert prefs.get_ambient_density() == ambient.DENSITY_RANGE[0]
    assert prefs.get_setting_animations_enabled() is False
    assert prefs.get_field_fade_enabled() is False


def test_leaving_extra_performance_gives_back_what_it_took():
    prefs.set_ambient_animation("cells")
    prefs.set_ambient_resolution(1.75)
    prefs.set_ambient_density(1.5)
    prefs.set_setting_animations_enabled(True)
    prefs.set_field_fade_enabled(True)
    wanted = prefs._visual_snapshot()

    prefs.set_spacr_mode("extra_performance")
    assert prefs._visual_snapshot() != wanted
    prefs.set_spacr_mode("balanced")
    assert prefs._visual_snapshot() == wanted


def test_re_entering_extra_performance_does_not_stash_the_minimums():
    """Otherwise leaving it would "restore" the minimums it just wrote."""
    prefs.set_ambient_animation("bokeh")
    wanted = prefs._visual_snapshot()
    prefs.set_spacr_mode("extra_performance")
    prefs.set_spacr_mode("extra_performance")
    prefs.set_spacr_mode("balanced")
    assert prefs._visual_snapshot() == wanted


def test_an_unreadable_stash_is_discarded_rather_than_guessed_at(
        _isolated_qsettings):
    prefs.set_spacr_mode("extra_performance")
    _isolated_qsettings.setValue("prefs/mode_visual_stash", "{not json")
    prefs.set_spacr_mode("balanced")
    # The minimums stand: they are visible and changeable, which is better
    # than being handed somebody's idea of a default.
    assert prefs.get_setting_animations_enabled() is False
    assert prefs.get_spacr_mode() == "balanced"


# ---------------------------------------------------------------------------
# The dialog
# ---------------------------------------------------------------------------

def test_the_dialog_offers_the_three_modes(qtbot, qt_theme_applied):
    from PySide6.QtWidgets import QComboBox
    from spacr.qt.preferences import PreferencesDialog

    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    combo = dlg.findChild(QComboBox, "SpacrMode")
    assert combo is not None
    keys = [combo.itemData(i) for i in range(combo.count())]
    assert keys == list(prefs.SPACR_MODES)
    assert combo.currentData() == "balanced"


def test_the_dialog_warns_on_selection_not_after_saving(qtbot,
                                                        qt_theme_applied):
    """A warning that arrives after the dialog closes is a report."""
    from PySide6.QtWidgets import QComboBox, QLabel
    from spacr.qt.preferences import PreferencesDialog

    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    combo = dlg.findChild(QComboBox, "SpacrMode")
    note = dlg.findChild(QLabel, "SpacrModeNote")
    keys = [combo.itemData(i) for i in range(combo.count())]

    combo.setCurrentIndex(keys.index("balanced"))
    assert "⚠" not in note.text()

    combo.setCurrentIndex(keys.index("extra_performance"))
    assert "⚠" in note.text()
    assert prefs.mode_warning("extra_performance")[:40] in note.text()
    assert prefs.get_spacr_mode() == "balanced", (
        "selecting a mode must not apply it; Save does that")

    combo.setCurrentIndex(keys.index("performance"))
    assert "⚠" in note.text()


def test_saving_applies_the_mode_after_the_visual_settings(qtbot,
                                                           qt_theme_applied):
    """Extra Performance overrides five of the settings the dialog writes.

    Written earlier, the dialog's own values would land on top of the
    minimums and the mode would silently not take effect.
    """
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox
    from spacr.qt.preferences import PreferencesDialog

    prefs.set_ambient_animation("cells")
    prefs.set_field_fade_enabled(True)

    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    combo = dlg.findChild(QComboBox, "SpacrMode")
    keys = [combo.itemData(i) for i in range(combo.count())]
    combo.setCurrentIndex(keys.index("extra_performance"))
    dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()

    assert prefs.get_spacr_mode() == "extra_performance"
    assert prefs.get_ambient_enabled() is False
    assert prefs.get_field_fade_enabled() is False
