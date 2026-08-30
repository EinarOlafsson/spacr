"""The seams of ``spacr.qt.app`` that only a wrong-shaped world reaches.

Three kinds of branch live here, none of them exercised by opening the
window normally:

* the REGISTRATION BLOCK that runs while ``spacr.qt.app`` is being imported
  -- a built-in module that declares no catalog row, and a plugin whose key
  collides with a built-in one. Neither can be provoked from an already
  imported module, so those tests re-execute the real module source in a
  private namespace, the way ``tests/qt/test_ambient_motion.py`` re-executes
  a shipped engine. Nothing is reloaded: the live ``spacr.qt.app`` and its
  registry are left exactly as they were found.
* the WINDOW's fall-through arms -- a wheel that carried no notch, a close
  the base handler refused, a seed for a screen that has no settings form.
* ``launch``'s one-off starts -- a machine that keeps crashing, a laptop, a
  benchmark run, a launch told not to ask its setup questions.
"""
from __future__ import annotations

import logging
import os
import pathlib
import sys
import types

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QCloseEvent, QFont, QWheelEvent
from PySide6.QtWidgets import QLineEdit, QWidget

import spacr.plugins as plugins
import spacr.qt.app_catalog as app_catalog
from spacr.qt import app as app_mod
from spacr.qt.app import MainWindow

# The launch harness is deliberately borrowed rather than re-invented: it is
# the one stand-in QApplication that survives `launch` installing an event
# filter parented to it.
from tests.qt.test_cov_qt_app import launched  # noqa: F401


@pytest.fixture
def win(qtbot, qt_theme_applied):
    """A live MainWindow, cleaned up by pytest-qt."""
    window = MainWindow()
    qtbot.addWidget(window)
    return window


# ---------------------------------------------------------------------------
# The registration block, run again against a world it never meets in one
# process: a module with no declared row, and a plugin that wants a taken key.
# ---------------------------------------------------------------------------

#: A key no built-in and no real plugin owns, so a row under it can only have
#: come from the contribution this file makes up.
FAKE_PLUGIN_KEY = "cov_r3_fake_plugin_app"

#: A built-in whose key the registry certainly already holds.
TAKEN_KEY = "mask"

#: The declared module whose row is hidden, so the loop falls through to
#: `import_module(...)` + `register()` for exactly one entry.
UNDECLARED_MODULE = "spacr.qt.screens.tabulate"


@pytest.fixture
def registration_run(monkeypatch):
    """Re-run ``spacr.qt.app``'s import-time registration in its own namespace.

    The block reads three things from outside itself -- ``declared_for``,
    ``plugin_apps`` and ``record_diagnostic`` -- and every one of them is
    replaced here, so the run sees a module with no declared row and two
    plugin apps, one of them colliding with a built-in.

    Everything it writes lands in the returned namespace's own ``APPS``:
    ``register_declared`` checks the LIVE registry before appending, so a key
    that is already registered is a no-op there, and the side tables
    ``_publish_meta`` pushes into are the ones ``conftest``'s autouse
    ``_restore_app_registry`` puts back after every test.
    """
    def _app(key, name, description):
        return plugins.AppContribution(key=key, name=name, entrypoint="",
                                       description=description, defaults="")

    collided = _app(TAKEN_KEY, "Impostor", "wants a taken key")
    accepted = _app(FAKE_PLUGIN_KEY, "Fake Plugin App", "nothing else's")
    diagnostics: list = []
    registered_the_old_way: list = []

    real_declared_for = app_catalog.declared_for
    monkeypatch.setattr(
        app_catalog, "declared_for",
        lambda module: (None if module == UNDECLARED_MODULE
                        else real_declared_for(module)))
    stub = types.ModuleType(UNDECLARED_MODULE)
    stub.register = lambda: registered_the_old_way.append(UNDECLARED_MODULE)
    monkeypatch.setitem(sys.modules, UNDECLARED_MODULE, stub)
    monkeypatch.setattr(plugins, "plugin_apps", lambda: (collided, accepted))
    monkeypatch.setattr(
        plugins, "record_diagnostic",
        lambda key, message, *a, **k: diagnostics.append((key, message)))

    source = pathlib.Path(app_mod.__file__).read_text(encoding="utf-8")
    namespace = {"__name__": "spacr.qt._app_registration_rerun",
                 "__file__": app_mod.__file__, "__package__": "spacr.qt"}
    exec(compile(source, app_mod.__file__, "exec"), namespace)
    return {"apps": namespace["APPS"], "diagnostics": diagnostics,
            "imported": registered_the_old_way}


def test_a_module_that_declares_no_row_is_imported_and_asked_to_register(
        registration_run):
    """A screen with no catalog row must still get onto Home.

    The catalog is an optimisation -- a row filled in from strings so the
    screen's pandas/sklearn imports wait until somebody opens it. A module
    that has not been converted has no row, and only the old path puts it in
    the registry: import it and call its ``register()``. Without that
    fall-through an unconverted screen vanishes from the sidebar silently.
    """
    assert registration_run["imported"] == [UNDECLARED_MODULE]


def test_a_plugin_cannot_take_a_key_a_builtin_already_owns(registration_run):
    """A contribution may add an app; it may never replace one.

    Two rows under one key means two identical sidebar entries and a coin
    toss over which screen a saved run reopens, so the built-in is kept and
    the plugin is told why in a diagnostic. The same run registers a
    non-colliding contribution, which proves the refusal is about the
    collision and not about plugins in general.
    """
    keys = [row[0] for row in registration_run["apps"]]
    assert keys.count(TAKEN_KEY) == 1
    assert FAKE_PLUGIN_KEY in keys
    assert [key for key, _ in registration_run["diagnostics"]] == [TAKEN_KEY]
    assert "collides" in registration_run["diagnostics"][0][1]


# ---------------------------------------------------------------------------
# The window's fall-through arms
# ---------------------------------------------------------------------------

class _RefusedClose(QCloseEvent):
    """A close event that stays un-accepted however it is handled.

    Stands in for a base ``closeEvent`` that vetoes the close -- the state
    ``MainWindow.closeEvent`` reads before it decides to end the session.
    """

    def isAccepted(self):
        return False


def test_only_an_accepted_close_ends_the_session(win, monkeypatch):
    """Closing the main window is the ONE thing that quits spaCR.

    ``quitOnLastWindowClosed`` is off precisely so a figure window closing
    cannot take the session with it, which makes this branch the whole exit
    path. It has to read the event: a refused close must leave the
    application running, or a veto turns into a quit and the user loses
    whatever the veto was protecting.
    """
    quits = []
    monkeypatch.setattr(
        app_mod, "QApplication",
        types.SimpleNamespace(
            instance=lambda: types.SimpleNamespace(
                quit=lambda: quits.append("quit"))))

    win.closeEvent(_RefusedClose())
    assert quits == []

    win.closeEvent(QCloseEvent())
    assert quits == ["quit"]


def test_a_wheel_that_carried_no_notch_is_handed_on(win, monkeypatch):
    """A scroll the backdrop did not use must keep travelling.

    The window steers the spaceout zoom with the wheel, but a trackpad's
    pixel-precise scroll arrives with an ``angleDelta`` under one notch. If
    the window accepted those anyway, a scroll gesture over the window
    chrome would be swallowed instead of reaching whatever wanted it.
    """
    nudges = []
    from spacr.qt.widgets import fractal_travel
    monkeypatch.setattr(fractal_travel, "nudge_zoom_rate",
                        lambda steps: nudges.append(steps) or 4.0)

    def _wheel(dy):
        return QWheelEvent(
            QPointF(5.0, 5.0), QPointF(5.0, 5.0), QPoint(0, 0), QPoint(0, dy),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase, False)

    below_a_notch = _wheel(40)
    win.wheelEvent(below_a_notch)
    assert nudges == []
    assert not below_a_notch.isAccepted()

    a_whole_notch = _wheel(120)
    win.wheelEvent(a_whole_notch)
    assert nudges == [1]
    assert a_whole_notch.isAccepted()


def test_the_cellpose_workbench_is_wired_to_the_window_that_built_it(
        win, monkeypatch):
    """Train Cellpose is built by hand so its two signals stay visible.

    The screen reports a failed training run and asks to be submitted to a
    cluster; both are the window's to answer, and both are connected here
    rather than inside a factory so ``tests/qt/test_all_module_smoke.py`` can
    still read them out of this method's bytecode. A screen built without
    them is a workbench whose Submit button does nothing.
    """
    submissions = []
    monkeypatch.setattr(
        win, "_on_remote_submit_requested",
        lambda key, settings: submissions.append((key, settings)))

    screen = win._build_screen("train_cellpose")

    from spacr.qt.screens.train_cellpose import CellposeWorkbenchScreen
    assert isinstance(screen, CellposeWorkbenchScreen)
    screen.remote_submit_requested.emit("train_cellpose", {"epochs": 3})
    assert submissions == [("train_cellpose", {"epochs": 3})]


def test_a_remote_submit_reaches_the_job_screen_and_survives_its_absence(
        win, monkeypatch):
    """"Run this on the cluster" carries the settings that were on screen.

    The point of the hand-off is that the user does not retype anything: the
    module they were configuring and its current values arrive in Distributed
    Jobs already filled in. And when that screen could not be built at all,
    the request has to end quietly -- a missing optional screen must not turn
    a button press into a traceback over the module the user was working in.
    """
    win._on_remote_submit_requested("mask", {"src": "/data/plate1"})
    screen = win._screens["distributed_jobs"]
    assert screen._settings_snapshot == {"src": "/data/plate1"}

    win._screens.pop("distributed_jobs")
    monkeypatch.setattr(win, "_on_nav_selected", lambda key: None)
    win._on_remote_submit_requested("mask", {"src": "/data/plate2"})
    assert "distributed_jobs" not in win._screens
    assert screen._settings_snapshot == {"src": "/data/plate1"}


def test_a_seed_only_lands_where_there_is_a_form_to_take_it(win, monkeypatch):
    """Loading a run's settings must not depend on the screen being a form.

    "Train CV" on the annotator and "load this run's settings" in Run History
    both push a seed at whatever screen the key resolves to, and some of them
    are not settings-driven at all. A seed for one has nowhere to go, so it is
    dropped rather than raised: the navigation the user asked for has already
    happened, and an exception here strands them on a half-opened page.
    """
    monkeypatch.setattr(win, "open_module", lambda key: key)
    landed: list = []
    real_apply = MainWindow._apply_seed_value
    monkeypatch.setattr(
        MainWindow, "_apply_seed_value",
        staticmethod(lambda w, value:
                     landed.append(value) or real_apply(w, value)))

    win._screens["cov_r3_formless"] = QWidget()
    win._on_train_requested("cov_r3_formless", {"nucleus_channel": "1"})
    assert landed == []

    field = QLineEdit()
    seeded = QWidget()
    seeded._settings_model = types.SimpleNamespace(
        _widgets={"nucleus_channel": field})
    win._screens["cov_r3_seeded"] = seeded
    win._on_train_requested("cov_r3_seeded",
                            {"nucleus_channel": "1", "absent_setting": 7})
    assert landed == ["1"]
    assert field.text() == "1"


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

def test_the_interface_font_keeps_whatever_size_the_platform_chose(qapp):
    """Applying Open Sans must change the family and nothing else.

    The font-scale preference is applied on top of the platform's own point
    size later in the launch. If ``_use_open_sans`` wrote a size of its own it
    would silently undo that, and everyone who had chosen larger text would get
    the default back on the next start. A platform font set in PIXELS reports
    no point size to carry over, and copying the -1 that stands for "unset"
    would be a font with no size at all.
    """
    app_mod._load_bundled_fonts()

    def _apply(existing):
        applied = []
        stub = types.SimpleNamespace(font=lambda: existing,
                                     setFont=applied.append)
        family = app_mod._use_open_sans(stub, weight="regular")
        return family, applied

    in_points = QFont("Whatever")
    in_points.setPointSizeF(17.5)
    family, applied = _apply(in_points)
    assert family == "Open Sans"
    assert applied[0].pointSizeF() == pytest.approx(17.5)

    in_pixels = QFont("Whatever")
    in_pixels.setPixelSize(13)
    assert in_pixels.pointSizeF() < 0
    family, applied = _apply(in_pixels)
    assert family == "Open Sans"
    untouched = QFont("Open Sans").pointSizeF()
    assert applied[0].pointSizeF() == pytest.approx(untouched)


def test_only_font_files_in_the_bundle_are_handed_to_qt(monkeypatch):
    """Everything in the bundled font directory is not a font.

    A licence file, a README or an editor's backup sits beside the TTFs in
    any real checkout, and ``QFontDatabase.addApplicationFont`` answers -1
    for each of them. Filtering by extension keeps that noise out of the
    font database; without it every non-font in the directory becomes a
    failed load at every launch.
    """
    import PySide6.QtGui as qtgui
    handed: list = []
    monkeypatch.setattr(
        qtgui, "QFontDatabase",
        types.SimpleNamespace(addApplicationFont=handed.append))
    listed = ["OFL.txt", "README.md", "OpenSans-Regular.ttf", "OpenSans.otf"]
    monkeypatch.setattr(
        app_mod, "os",
        types.SimpleNamespace(path=os.path, listdir=lambda d: list(listed)))

    app_mod._load_bundled_fonts()

    assert [os.path.basename(p) for p in handed] == [
        "OpenSans-Regular.ttf", "OpenSans.otf"]


# ---------------------------------------------------------------------------
# launch()
# ---------------------------------------------------------------------------

def test_a_run_of_unclean_exits_starts_without_the_backdrop(
        launched, monkeypatch):
    """Two crashes in a row and the animated background is left out.

    The setting that would turn the backdrop off lives behind the window
    that never appears, so a driver that dies on the spaceout fractal is a
    loop the user cannot break from inside spaCR. The launch that reads the
    count has to actually make the call that turns GL off -- counting and then
    drawing anyway is the same loop with extra logging.
    """
    from spacr.qt import crash_recovery
    dropped = []
    monkeypatch.setattr(crash_recovery, "should_start_without_the_backdrop",
                        lambda unclean=None: True)
    monkeypatch.setattr(crash_recovery, "take_the_backdrop_out_of_this_launch",
                        lambda: dropped.append("dropped"))

    assert app_mod.launch([]) == 0
    assert dropped == ["dropped"]


def test_a_launch_told_to_skip_the_setup_never_opens_it(
        launched, monkeypatch):
    """``--no-setup`` is what a batch job passes, and it has to be obeyed.

    The setup screen is modal and it is the first thing a launch draws. On a
    server that inherits a stale profile, opening it means a job sitting on an
    invisible dialog until something kills it, with a run that never starts as
    the only symptom.
    """
    from spacr.qt.widgets import setup_slides
    asked = []
    monkeypatch.setattr(setup_slides, "open_setup_if_needed",
                        lambda parent: asked.append(parent))

    assert app_mod.launch(["--no-setup"]) == 0
    assert asked == []

    assert app_mod.launch([]) == 0
    assert asked == [None]


def test_the_setup_answers_are_applied_before_the_window_is_built(
        launched, monkeypatch):
    """An answer given at setup decides how the main window is BUILT.

    Language, theme and font scale are read while the window is assembled, so
    answers arriving after it exists mean a half-built window on screen in the
    wrong language, restyled under the user as they read it. Preferences are
    applied a second time -- and only when there were answers, because
    re-resolving the theme on every later launch is work nobody asked for.
    """
    from spacr.qt import preferences
    from spacr.qt.widgets import setup_slides
    real_apply = preferences.apply_preferences_to_app
    applied: list = []

    def _apply(app):
        applied.append(app)
        return real_apply(app)

    monkeypatch.setattr(preferences, "apply_preferences_to_app", _apply)
    monkeypatch.setattr(setup_slides, "open_setup_if_needed",
                        lambda parent: None)
    assert app_mod.launch([]) == 0
    once = len(applied)
    assert once == 1

    monkeypatch.setattr(setup_slides, "open_setup_if_needed",
                        lambda parent: {"language": "en"})
    assert app_mod.launch([]) == 0
    assert len(applied) == once + 2
    assert applied[-1] is applied[-2]


def test_matplotlib_is_pinned_to_agg_before_it_is_ever_imported(
        launched, monkeypatch):
    """A matplotlib canvas built off the GUI thread kills the process.

    So the backend is not the caller's to choose. When matplotlib has not been
    imported yet, the cheapest way to pin it is the environment variable it
    reads for itself at import: calling ``matplotlib.use`` here would drag the
    whole package into a launch that has not drawn anything yet.
    """
    monkeypatch.setenv("MPLBACKEND", "TkAgg")
    monkeypatch.delitem(sys.modules, "matplotlib", raising=False)

    assert app_mod.launch([]) == 0
    assert os.environ["MPLBACKEND"] == "Agg"


def test_laptop_mode_says_what_it_turned_down(launched, monkeypatch, caplog):
    """A launch that quietly turns things off has to say which things.

    Laptop mode dims decoration, never analysis, so a run computes the same
    answer either way -- which is what makes deciding it automatically
    acceptable. The log line is the only place the decision is visible, and
    without it "it looks different on my laptop" has no evidence behind it.
    """
    from spacr.qt import laptop_mode
    monkeypatch.setattr(laptop_mode, "apply",
                        lambda: {"changed": ["backdrop", "shadows"]})

    with caplog.at_level(logging.INFO, logger="spacr.qt.app"):
        assert app_mod.launch([]) == 0
    assert "laptop mode changed: backdrop, shadows" in caplog.text


def test_a_benchmark_run_gets_its_controller_and_keeps_it(
        launched, monkeypatch, tmp_path):
    """``SPACR_BENCHMARK_JSON`` turns a launch into an unattended run.

    The controller is what ends that run and writes the measurements; it is
    only built when the variable asks for it, and it is held on the window
    afterwards because losing the Python reference to it is losing the run.
    """
    from spacr.qt import startup_benchmark
    controller = object()
    started = []
    monkeypatch.setenv("SPACR_BENCHMARK_JSON", str(tmp_path / "bench.json"))
    monkeypatch.setattr(
        startup_benchmark, "maybe_start",
        lambda app, win: started.append((app, win)) or controller)

    assert app_mod.launch([]) == 0
    window = launched["window"]()
    assert [pair[1] for pair in started] == [window]
    assert window._startup_benchmark_controller is controller
