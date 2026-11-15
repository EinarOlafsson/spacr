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

import pathlib
import sys
import types

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QFont, QWheelEvent

import spacr.plugins as plugins
import spacr.qt.app_catalog as app_catalog
from spacr.qt import app as app_mod
from spacr.qt.app import MainWindow

# The launch harness is deliberately borrowed rather than re-invented: it is
# the one stand-in QApplication that survives `launch` installing an event
# filter parented to it.
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


# ---------------------------------------------------------------------------
# launch()
# ---------------------------------------------------------------------------

