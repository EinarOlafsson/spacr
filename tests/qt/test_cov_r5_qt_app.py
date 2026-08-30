"""What ``spacr.qt.app`` still does when a collaborator lets it down.

Every arc here is a recovery arm, and each one guards the same promise
from a different side: a module opens, and the window survives, even when
the part that was supposed to help it fails.

* REBUILDING A FORM. A committed channel value rebuilds the whole screen.
  The old screen may carry no settings model to copy the organelle presets
  from, the build itself may raise, the three decoration passes may each
  fail, and retiring the screen that was replaced may fail after the
  replacement is already on the stack.
* OPENING A MODULE. The preparing card may not be makeable and the card
  that exists may already be dead by the time it is taken down; the plugin
  registry may be unreadable; the help pass may throw; the ambient backdrop
  may refuse to install; a seed the caller pushed may be rejected by the
  screen that took it.
* COMING BACK UP. A force-restart record that names a module but carries no
  settings still reopens the module.
* LAUNCHING. Pinning matplotlib, installing the thread guard, opening the
  setup screen, draining the job runners and recording a clean shutdown are
  each wrapped, because none of them is worth a launch; and a timing report
  that was written has to say where it went.

Nothing here asserts a bare "it did not raise": every test drives the
working path in the same body, so the failure arm is measured against what
success looks like.
"""
from __future__ import annotations

import logging
import sys
import types

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QFont, QWheelEvent
from PySide6.QtWidgets import QLabel, QWidget

import spacr.plugins as plugins
import spacr.restart_state as restart_state
from spacr.qt import app as app_mod
from spacr.qt import i18n as i18n_mod
from spacr.qt import preferences as preferences_mod
from spacr.qt.app import MainWindow
from spacr.qt.screens import app_screen as app_screen_mod
from spacr.qt.screens import settings_model as settings_model_mod
from spacr.qt.widgets import ambient as ambient_mod

# The launch harness is borrowed rather than re-invented: it is the one
# stand-in QApplication that survives `launch` installing an event filter
# parented to it, and the only one whose `exec` returns.
from tests.qt.test_cov_qt_app import launched  # noqa: F401

#: A key no built-in and no plugin owns, so anything registered under it
#: came from this file. The registry is put back by ``tests/qt/conftest``'s
#: autouse ``_restore_app_registry`` after every test.
PROBE = "cov_r5_probe"


@pytest.fixture
def win(qtbot, qt_theme_applied):
    """A live MainWindow, cleaned up by pytest-qt."""
    window = MainWindow()
    qtbot.addWidget(window)
    return window


class _Screen(QWidget):
    """A module screen that records what the window asks of it.

    Stands in for a real ``AppScreen`` in the two places the window treats
    a screen as a collaborator rather than as a widget: publishing its
    workspace when it is swapped in, and taking a seed pushed at it.
    """

    def __init__(self, boom=None):
        super().__init__()
        self.boom = boom if boom is not None else {}
        self.workspaces = 0
        self.seeds: list = []
        self.settings: list = []
        self.built_for = None

    def register_workspace(self):
        self.workspaces += 1
        if self.boom.get("workspace"):
            raise RuntimeError("the workspace store is gone")

    def apply_seed(self, seed):
        self.seeds.append(seed)
        if self.boom.get("seed"):
            raise ValueError("this screen cannot take that seed")

    def apply_settings_dict(self, settings):
        self.settings.append(dict(settings))
        return len(settings)


@pytest.fixture
def probe_app():
    """Register :data:`PROBE` with a factory this test file controls.

    Returns the shared state: ``boom`` switches a collaborator's failure on
    and off inside one test, ``built`` collects every screen the window
    made, and ``fail_build`` makes the factory itself raise.
    """
    state = {"boom": {}, "built": [], "fail_build": False}

    def factory():
        if state["fail_build"]:
            raise RuntimeError("this screen cannot be built today")
        screen = _Screen(state["boom"])
        screen.built_for = dict(
            app_screen_mod.AppScreen.values_the_next_screen_is_built_for or {})
        state["built"].append(screen)
        return screen

    app_mod.register_app(PROBE, "Probe", "a screen this file owns",
                         app_mod.SECTION_EXPLORE, factory=factory)
    return state


# ---------------------------------------------------------------------------
# The wheel
# ---------------------------------------------------------------------------

class _UnreadableWheel(QWheelEvent):
    """A wheel event whose delta cannot be read.

    A real ``QWheelEvent`` subclass, not a duck: the fall-through hands the
    event to ``QMainWindow.wheelEvent``, which is C++ and takes nothing
    else. Only the Python-side ``angleDelta`` call is overridden, which is
    the one the window makes.
    """

    def angleDelta(self):
        raise RuntimeError("the event's delta could not be read")


def test_a_wheel_whose_delta_cannot_be_read_steers_nothing(win, monkeypatch):
    """An unreadable notch count must be zero, not a guess.

    The wheel drives the spaceout zoom, and the delta is read from an event
    the toolkit owns. Letting the read fail loudly would take the window
    down on a scroll gesture; letting it fall through to a stale or partial
    number would move the backdrop by an amount nobody asked for. Zero is
    the only honest answer, and zero means the scroll travels on.
    """
    steers: list = []
    monkeypatch.setattr(MainWindow, "_steer_the_backdrop",
                        lambda self, steps: steers.append(steps) or True)

    unreadable = _UnreadableWheel(
        QPointF(5.0, 5.0), QPointF(5.0, 5.0), QPoint(0, 0), QPoint(0, 240),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False)
    win.wheelEvent(unreadable)
    assert steers == []
    assert not unreadable.isAccepted()

    # The same gesture with a delta that CAN be read: two notches, steered,
    # and swallowed so it reaches nothing behind the window.
    readable = QWheelEvent(
        QPointF(5.0, 5.0), QPointF(5.0, 5.0), QPoint(0, 0), QPoint(0, 240),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False)
    win.wheelEvent(readable)
    assert steers == [2]
    assert readable.isAccepted()


# ---------------------------------------------------------------------------
# rebuild_app_screen
# ---------------------------------------------------------------------------

def test_a_rebuild_that_cannot_build_leaves_the_screen_that_works(
        win, probe_app, monkeypatch, caplog):
    """A failed rebuild must not take the working screen away with it.

    The rebuild exists to change a form's SHAPE, so it runs on a screen the
    user is looking at. If the build raises and the old screen has already
    been dropped, a committed channel value empties the window. It is also
    where the build-values hand-off has to be undone: that class attribute
    shapes whatever screen is built next, and a value left behind after a
    failure would silently seed a different module.

    The old screen here carries no settings model at all -- a plain page
    such as Annotate or the Database Browser -- so the organelle presets
    cannot be copied off it either.
    """
    sentinel = {"left_over": "from an earlier build"}
    monkeypatch.setattr(app_screen_mod.AppScreen,
                        "values_the_next_screen_is_built_for", sentinel)
    old = QWidget()
    assert not hasattr(old, "_settings_model")
    win._screens[PROBE] = old
    win._stack.addWidget(old)

    probe_app["fail_build"] = True
    with caplog.at_level(logging.ERROR, logger="spacr.qt.app"):
        win.rebuild_app_screen(PROBE, {"nucleus_channel": 2})

    assert f"could not rebuild the {PROBE} screen" in caplog.text
    assert win._screens[PROBE] is old
    assert probe_app["built"] == []
    assert (app_screen_mod.AppScreen.values_the_next_screen_is_built_for
            is sentinel)

    # The same call with a factory that works: the replacement is built
    # from the values it was given, and it takes the old screen's place.
    probe_app["fail_build"] = False
    win.rebuild_app_screen(PROBE, {"nucleus_channel": 2})
    fresh = probe_app["built"][-1]
    assert fresh.built_for == {"nucleus_channel": 2}
    assert win._screens[PROBE] is fresh
    assert win._stack.currentWidget() is fresh
    assert (app_screen_mod.AppScreen.values_the_next_screen_is_built_for
            is sentinel)


def test_a_rebuilt_screen_is_shown_even_when_every_dressing_pass_fails(
        win, probe_app, monkeypatch, caplog):
    """Theme, translation and help are decoration; the screen is the point.

    All three run over the replacement before it goes on the stack, and all
    three touch code the module does not own -- a stylesheet, a translation
    catalogue, a tooltip table. Any of them raising must cost only itself:
    a user who committed a channel value gets their form back in English,
    or unstyled, or without its help, rather than not at all.
    """
    boom = {"on": True}

    def _raise_when_on(*args, **kwargs):
        if boom["on"]:
            raise RuntimeError("the dressing pass failed")
        return args[0] if args else None

    themed: list = []
    translated: list = []
    retargeted: list = []

    def _theme(self, screen, key):
        themed.append(screen)
        _raise_when_on()

    monkeypatch.setattr(MainWindow, "_theme_screen", _theme)
    monkeypatch.setattr(i18n_mod, "retranslate_widget_tree",
                        lambda w: translated.append(w) or _raise_when_on())
    monkeypatch.setattr(settings_model_mod, "retarget_field_tooltips",
                        lambda w: retargeted.append(w) or _raise_when_on())

    with caplog.at_level(logging.ERROR, logger="spacr.qt.app"):
        win.rebuild_app_screen(PROBE, {})

    fresh = probe_app["built"][-1]
    assert [themed[-1], translated[-1], retargeted[-1]] == [fresh] * 3
    assert f"Could not theme the rebuilt {PROBE} screen" in caplog.text
    assert f"Could not translate the rebuilt {PROBE} screen" in caplog.text
    assert f"Could not retarget help on the {PROBE} screen" in caplog.text
    assert win._screens[PROBE] is fresh
    assert win._stack.currentWidget() is fresh

    # And with the same three passes working, the replacement is dressed
    # and shown -- the failure arms above cost the dressing, nothing else.
    boom["on"] = False
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="spacr.qt.app"):
        win.rebuild_app_screen(PROBE, {})
    second = probe_app["built"][-1]
    assert second is not fresh
    assert [themed[-1], translated[-1], retargeted[-1]] == [second] * 3
    assert "rebuilt" not in caplog.text
    assert win._stack.currentWidget() is second


def test_a_replacement_stays_up_when_the_old_screen_will_not_retire(
        win, probe_app, monkeypatch, caplog):
    """Retiring the screen that was replaced happens AFTER the swap.

    The replacement is built and shown first, on purpose: removing the old
    one first drops the window to Home. That ordering means a failure while
    the old screen is being torn down finds the new screen already on
    screen, so the recovery is to leave it there and re-publish its
    workspace -- not to unwind a swap the user has already seen.

    Here even that second publish fails, which is the last arm: a window
    with a live screen and no workspace entry is still a window the user
    can work in.
    """
    old = QWidget()
    win._screens[PROBE] = old
    win._stack.addWidget(old)
    probe_app["boom"]["workspace"] = True

    with caplog.at_level(logging.DEBUG, logger="spacr.qt.app"):
        win.rebuild_app_screen(PROBE, {})

    fresh = probe_app["built"][-1]
    assert fresh.workspaces == 2, "the failed publish must be retried once"
    assert f"could not retire the {PROBE} screen" in caplog.text
    assert f"could not restore {PROBE} workspace" in caplog.text
    assert win._screens[PROBE] is fresh
    assert win._stack.currentWidget() is fresh
    assert win._stack.indexOf(old) != -1, "a failed retire left old in place"

    # With a workspace that accepts the publish, the old screen really is
    # taken off the stack and the replacement is published exactly once.
    probe_app["boom"]["workspace"] = False
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="spacr.qt.app"):
        win.rebuild_app_screen(PROBE, {})
    second = probe_app["built"][-1]
    assert second.workspaces == 1
    assert win._stack.indexOf(fresh) == -1
    assert f"could not retire the {PROBE} screen" not in caplog.text


# ---------------------------------------------------------------------------
# Opening a module
# ---------------------------------------------------------------------------

def test_a_module_opens_even_when_its_help_cannot_be_moved(
        win, probe_app, monkeypatch, caplog):
    """Tooltips are re-aimed on every navigation, for every screen.

    It is the one place all screens pass through, which also makes it the
    one place a single broken tooltip table could stop every module from
    opening. Help in the wrong place is a blemish; a module that will not
    open is not.
    """
    seen: list = []
    boom = {"on": True}

    def _retarget(widget):
        seen.append(widget)
        if boom["on"]:
            raise RuntimeError("the tooltip table is unreadable")

    monkeypatch.setattr(settings_model_mod, "retarget_field_tooltips",
                        _retarget)

    with caplog.at_level(logging.ERROR, logger="spacr.qt.app"):
        win._on_nav_selected(PROBE)

    screen = probe_app["built"][-1]
    assert seen == [screen]
    assert f"Could not retarget help on the {PROBE} screen" in caplog.text
    assert win._stack.currentWidget() is screen
    assert win._status_app_label.text() == "Probe"

    # A second visit with a working table re-aims the help on the SAME
    # screen -- the pass runs on every navigation, not only on the build.
    boom["on"] = False
    win._on_nav_selected("__home__")
    win._on_nav_selected(PROBE)
    assert seen == [screen, screen]
    assert win._stack.currentWidget() is screen


def test_a_backdrop_that_will_not_install_costs_only_the_backdrop(
        win, monkeypatch, caplog):
    """The ambient artwork is the most optional thing on a screen.

    It is also the thing most likely to fail: it reads four preferences and
    then asks a GL-backed widget to attach itself. A screen with no
    animation behind it is a screen; a module that refuses to open because
    its wallpaper failed is a bug.
    """
    installed: list = []
    boom = {"on": True}

    def _install(screen, _second, **kwargs):
        installed.append((screen, kwargs.get("theme")))
        if boom["on"]:
            raise RuntimeError("no GL context for the backdrop")

    monkeypatch.setattr(preferences_mod, "get_ambient_enabled", lambda: True)
    monkeypatch.setattr(preferences_mod, "get_ambient_theme", lambda: "aurora")
    monkeypatch.setattr(ambient_mod, "install_ambient", _install)

    screen = QWidget()
    with caplog.at_level(logging.ERROR, logger="spacr.qt.app"):
        win._theme_screen(screen, PROBE)
    assert installed == [(screen, "aurora")]
    assert f"Could not install the backdrop for {PROBE}" in caplog.text

    boom["on"] = False
    caplog.clear()
    other = QWidget()
    with caplog.at_level(logging.ERROR, logger="spacr.qt.app"):
        win._theme_screen(other, PROBE)
    assert installed[-1] == (other, "aurora")
    assert "Could not install the backdrop" not in caplog.text


def test_a_preparing_card_that_cannot_be_made_is_simply_absent(
        win, monkeypatch, caplog):
    """The card is what a user looks at while a form is built.

    Its whole reason to exist is that the build blocks the GUI thread, so
    it must never be able to block it further -- including by raising on
    the way up. ``None`` is a complete answer: the module opens with no
    card rather than not at all.
    """
    with caplog.at_level(logging.DEBUG, logger="spacr.qt.app"):
        card = win._show_preparing(PROBE)
    assert isinstance(card, QLabel)
    assert card.text().endswith(f"{PROBE}…")
    assert card.parent() is win
    # `isVisible` answers for the whole ancestry and this window is never
    # shown; `isHidden` is the flag the card's own `show()` cleared.
    assert not card.isHidden()

    # `registered_metadata` is the first thing the card reads; a registry
    # in the middle of a mutation is what makes it throw.
    def _unreadable(_field):
        raise RuntimeError("the app registry is mid-write")

    monkeypatch.setattr(app_mod, "registered_metadata", _unreadable)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="spacr.qt.app"):
        assert win._show_preparing(PROBE) is None
    assert "could not show the preparing card" in caplog.text


def test_a_card_that_is_already_gone_is_still_taken_down(win):
    """Hiding the card runs in a ``finally``, after a build that may have died.

    By then the card may be a Python wrapper around a C++ object Qt has
    already deleted, and touching one of those raises. That exception would
    replace whatever the build was reporting -- the real failure -- with a
    complaint about a label.
    """
    real = win._show_preparing(PROBE)
    assert not real.isHidden()
    win._hide_preparing(real)
    assert real.isHidden()

    calls: list = []

    class _Dead:
        def hide(self):
            calls.append("hide")
            raise RuntimeError("wrapped C/C++ object has been deleted")

        def deleteLater(self):
            calls.append("deleteLater")

    win._hide_preparing(_Dead())
    assert calls == ["hide"], "the delete cannot run once hide has failed"


def test_a_module_opens_when_the_plugin_registry_cannot_be_read(
        win, probe_app, monkeypatch, caplog):
    """Every build asks the plugin registry whether a plugin owns the key.

    That registry is third-party data on disk, so reading it is the one
    step in the build that an outside package can break. A built-in module
    must not go down with it: the answer defaults to "no plugin owns this"
    and the built-in factory runs.
    """
    monkeypatch.setattr(
        plugins, "get_app",
        lambda key: (_ for _ in ()).throw(RuntimeError("plugin index gone")))

    with caplog.at_level(logging.ERROR, logger="spacr.qt.app"):
        screen = win._build_screen(PROBE)
    assert screen is probe_app["built"][-1]
    assert f"Could not inspect plugin screen contribution {PROBE}" \
        in caplog.text

    # And with a readable registry that claims no owner, the same built-in
    # factory runs with nothing logged.
    monkeypatch.setattr(plugins, "get_app", lambda key: None)
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="spacr.qt.app"):
        again = win._build_screen(PROBE)
    assert again is probe_app["built"][-1] and again is not screen
    assert "Could not inspect plugin screen contribution" not in caplog.text


def test_a_seed_a_screen_refuses_is_reported_not_swallowed(
        win, probe_app, monkeypatch, caplog):
    """A screen that takes its own seed decides what a seed means.

    Train CV, Run History's "load this run's settings" and the hit-list
    hand-off all push a dict at a screen the caller did not write. When the
    screen rejects it, the navigation has already happened and the user is
    looking at the module -- so the failure belongs in the log with the
    seed in it, not on top of a window that is otherwise fine.
    """
    probe_app["boom"]["seed"] = True
    with caplog.at_level(logging.WARNING, logger="spacr.qt.app"):
        win._on_train_requested(PROBE, {"target_gene": "PTEN"})

    screen = probe_app["built"][-1]
    assert screen.seeds == [{"target_gene": "PTEN"}]
    assert f"Could not seed {PROBE}" in caplog.text
    assert "PTEN" in caplog.text
    assert win._stack.currentWidget() is screen

    # The same push at a screen that accepts it: taken, and nothing logged.
    probe_app["boom"]["seed"] = False
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="spacr.qt.app"):
        win._on_train_requested(PROBE, {"target_gene": "TP53"})
    assert screen.seeds[-1] == {"target_gene": "TP53"}
    assert f"Could not seed {PROBE}" not in caplog.text


def test_a_restart_record_with_no_settings_still_reopens_the_module(
        win, probe_app, monkeypatch):
    """The module is the point of the record; the settings are a bonus.

    A Force restart writes whatever the open screen could give it, and a
    screen with no settings model gives nothing. Reopening only when there
    are settings to apply would mean the restart dropped the user back on
    Home, which is exactly the state the record exists to avoid.
    """
    monkeypatch.setattr(restart_state, "take", lambda: {"module": PROBE})
    assert win.resume_after_restart() == PROBE
    screen = probe_app["built"][-1]
    assert win._stack.currentWidget() is screen
    assert screen.settings == []

    # A record that DOES carry settings applies them to the screen it
    # reopens -- same path, one more step.
    win._screens.pop(PROBE)
    monkeypatch.setattr(
        restart_state, "take",
        lambda: {"module": PROBE, "settings": {"src": "/data/plate1"}})
    assert win.resume_after_restart() == PROBE
    reopened = probe_app["built"][-1]
    assert reopened is not screen
    assert reopened.settings == [{"src": "/data/plate1"}]


# ---------------------------------------------------------------------------
# The interface font
# ---------------------------------------------------------------------------

def test_an_unreadable_font_preference_falls_back_to_regular(monkeypatch):
    """The interface font weight is a preference, and preferences can fail.

    Reading it means a settings store on disk. Falling back to Light on a
    failure would be a readability regression nobody chose; falling back to
    Regular is the shipped default, which is what a profile that has never
    been asked already gets.
    """
    app_mod._load_bundled_fonts()
    applied: list = []
    stub = types.SimpleNamespace(font=lambda: QFont("Whatever"),
                                 setFont=applied.append)

    monkeypatch.setattr(
        preferences_mod, "get_interface_font_weight",
        lambda: (_ for _ in ()).throw(OSError("the preference store is gone")))
    assert app_mod._use_open_sans(stub) == "Open Sans"
    assert applied[-1].weight() == QFont.Weight.Normal

    # The preference is genuinely consulted: a readable "light" is obeyed,
    # which is what makes the fallback above a fallback and not the rule.
    monkeypatch.setattr(preferences_mod, "get_interface_font_weight",
                        lambda: "light")
    assert app_mod._use_open_sans(stub) == "Open Sans"
    assert applied[-1].weight() == QFont.Weight.Light


# ---------------------------------------------------------------------------
# launch()
# ---------------------------------------------------------------------------

def test_a_matplotlib_that_refuses_the_agg_backend_does_not_stop_the_launch(
        launched, monkeypatch):
    """A Qt canvas built on a worker thread kills the process.

    So the backend is forced, and when matplotlib is ALREADY imported the
    environment variable is too late -- the backend was read at its import
    -- and only ``use(force=True)`` can still move it. That call goes into
    a package the launch does not own, and a launch that died because a
    third-party backend switch raised would be a spaCR that will not start
    on somebody else's matplotlib.
    """
    import matplotlib

    assert "matplotlib" in sys.modules
    used: list = []
    monkeypatch.setattr(
        matplotlib, "use",
        lambda backend, force=False: used.append((backend, force)))
    assert app_mod.launch([]) == 0
    assert used == [("Agg", True)]

    monkeypatch.setattr(
        matplotlib, "use",
        lambda *a, **k: (_ for _ in ()).throw(
            ImportError("no Agg backend in this build")))
    assert app_mod.launch([]) == 0


def test_a_thread_guard_that_will_not_install_does_not_stop_the_launch(
        launched, monkeypatch):
    """The guard names the caller of an off-thread timer start.

    It is instrumentation: it makes a crash report readable, and it is
    installed on every launch because the crash it explains only happens
    during a real run. Instrumentation that can prevent a start is worse
    than no instrumentation.
    """
    from spacr.qt import thread_guard

    installs: list = []
    monkeypatch.setattr(thread_guard, "install",
                        lambda: installs.append("installed"))
    assert app_mod.launch([]) == 0
    assert installs == ["installed"]

    monkeypatch.setattr(
        thread_guard, "install",
        lambda: (_ for _ in ()).throw(RuntimeError("cannot patch QTimer")))
    assert app_mod.launch([]) == 0
    assert installs == ["installed"], "a failed install must not be recorded"


def test_a_setup_screen_that_raises_is_not_worth_a_launch(
        launched, monkeypatch, caplog):
    """Every question the setup screen asks has a working default.

    Which makes it the most droppable step in the launch: a user who never
    sees it is exactly where a user who dismissed it would be. It is also
    the FIRST thing drawn, so an exception there is an exception before any
    window exists -- a spaCR that shows nothing at all.
    """
    from spacr.qt.widgets import setup_slides

    monkeypatch.setattr(
        setup_slides, "open_setup_if_needed",
        lambda parent: (_ for _ in ()).throw(RuntimeError("no slides")))
    with caplog.at_level(logging.DEBUG, logger="spacr.qt.app"):
        assert app_mod.launch([]) == 0
    assert "could not open the setup screen" in caplog.text

    window = launched["window"]()
    assert window.isVisible(), "the window is built whatever setup did"


def test_a_job_runner_registry_that_will_not_drain_still_lets_the_consoles_go(
        launched, monkeypatch, caplog):
    """Quitting with a job in flight is what aborts the process.

    ``aboutToQuit`` drains the runners and then every console panel. The
    two are independent, and the drain that runs first is the one reaching
    into a registry of live threads -- so if it throws, the consoles it was
    standing in front of must still be shut down, or the crash it exists to
    prevent happens anyway.
    """
    from spacr.qt import job_runner

    drained: list = []
    monkeypatch.setattr(job_runner, "shutdown_all",
                        lambda: drained.append("drained"))
    assert app_mod.launch([]) == 0
    quit_handlers = launched["shims"][0].aboutToQuit.callbacks
    assert len(quit_handlers) == 1
    quit_handlers[0]()
    assert drained == ["drained"]

    monkeypatch.setattr(
        job_runner, "shutdown_all",
        lambda: (_ for _ in ()).throw(RuntimeError("the runner list is gone")))
    consoles: list = []
    from spacr.qt.widgets import console_panel

    monkeypatch.setattr(console_panel.ConsolePanel, "shutdown",
                        lambda self: consoles.append(self))
    window = launched["window"]()
    panels = window.findChildren(console_panel.ConsolePanel)
    with caplog.at_level(logging.DEBUG, logger="spacr.qt.app"):
        quit_handlers[0]()
    assert "Could not drain the job runners" in caplog.text
    assert consoles == panels


def test_a_clean_shutdown_that_cannot_be_recorded_keeps_the_exit_code(
        launched, monkeypatch, caplog):
    """Returning from ``exec`` is the definition of a clean run.

    The count it clears is what decides whether the NEXT launch turns the
    backdrop off, and it is written to a file in the user's profile. A
    profile that is read-only would otherwise make the last act of a
    perfectly good session an exception, thrown after the event loop has
    ended, on top of the exit code the caller is waiting for.
    """
    from spacr.qt import crash_recovery

    noted: list = []
    monkeypatch.setattr(crash_recovery, "note_a_clean_shutdown",
                        lambda: noted.append("noted"))
    assert app_mod.launch([]) == 0
    assert noted == ["noted"]

    monkeypatch.setattr(
        crash_recovery, "note_a_clean_shutdown",
        lambda: (_ for _ in ()).throw(OSError("read-only profile")))
    with caplog.at_level(logging.DEBUG, logger="spacr.qt.app"):
        assert app_mod.launch([]) == 0
    assert "could not record a clean shutdown" in caplog.text


def test_a_timing_report_that_was_written_says_where_it_went(
        launched, monkeypatch, capsys, tmp_path):
    """A report nobody can find is a report that was not written.

    The path is chosen by the environment or falls back to the working
    directory, so the one line printed on the way out is the only place a
    user learns which. It is printed from a ``finally``, so it survives
    however the event loop ended.

    ``write_report`` is replaced rather than switching ``timing.ENABLED``
    on: that flag also installs a process-wide import hook and a GUI
    watchdog timer, both of which would outlive this test.
    """
    target = tmp_path / "spacr-timing.log"

    def _write():
        target.write_text("spaCR timing report\n")
        return str(target)

    monkeypatch.setattr(app_mod._timing, "write_report", _write)
    assert app_mod.launch([]) == 0
    printed = capsys.readouterr().out
    assert f"spaCR timing report written to {target}" in printed
    assert target.read_text().startswith("spaCR timing report")

    # Timing off: nothing is written, so nothing is announced.
    monkeypatch.setattr(app_mod._timing, "write_report", lambda: "")
    assert app_mod.launch([]) == 0
    assert "timing report written" not in capsys.readouterr().out
