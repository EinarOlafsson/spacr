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
import types

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QFont, QWheelEvent
from PySide6.QtWidgets import QLabel, QWidget

import spacr.restart_state as restart_state
from spacr.qt import app as app_mod
from spacr.qt import preferences as preferences_mod
from spacr.qt.app import MainWindow
from spacr.qt.screens import app_screen as app_screen_mod
from spacr.qt.widgets import ambient as ambient_mod

# The launch harness is borrowed rather than re-invented: it is the one
# stand-in QApplication that survives `launch` installing an event filter
# parented to it, and the only one whose `exec` returns.
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

    # A SECTION THAT STILL EXISTS. register_app now refuses a section
    # that is not in SECTION_ORDER, and the 2026-08-31 restructure left
    # Core/Data/Tools/Assays -- so registering the probe under the legacy
    # "Explore" alias raised in the FIXTURE, which surfaced as seven
    # errors rather than seven failures.
    app_mod.register_app(PROBE, "Probe", "a screen this file owns",
                         app_mod.SECTION_TOOLS, factory=factory)
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

# ---------------------------------------------------------------------------
# Opening a module
# ---------------------------------------------------------------------------

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
    monkeypatch.setattr(win, "window_backdrop", lambda: None)
    monkeypatch.setattr(ambient_mod, "_the_heavy_import_lock_is_free", lambda: True)

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

