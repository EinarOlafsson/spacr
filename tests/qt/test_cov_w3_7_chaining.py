"""The chaining strip's edges: the seams it does not find, and the ones that throw.

``tests/qt/test_chaining_gui.py`` drives the strip on real ``AppScreen``s and
holds it to what the user sees — the folder Mask registered, the pin that
survives a restart, the successor buttons. What is exercised here is the rest
of it: a host screen that is missing each seam in turn, and the failure of
every call the strip makes into one, because the promise this widget makes is
that it is an *aid* and can never take a module screen down with it.

The strip is built on a stand-in host rather than an ``AppScreen`` — the seams
it reaches for (``app_key``, ``_settings_model``, ``_btn_run``, ``_worker``,
``apply_settings_dict``) are few and named, and a host that offers a chosen
subset of them is the only way to ask what happens when one is absent.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QLineEdit, QWidget

from spacr import chaining as core_chaining
from spacr import ports as core_ports
from spacr.chaining import NextStep
from spacr.qt import chaining as qt_chaining

from spacr.qt.chaining import ChainingBar


@pytest.fixture(autouse=True)
def _own_pins(monkeypatch, tmp_path):
    """Never read or write the developer's real pin file."""
    monkeypatch.setenv(core_chaining.PIN_STATE_ENV, str(tmp_path / "pins.json"))
    core_chaining.pin_store(refresh=True)
    yield
    core_chaining.pin_store(refresh=True)


@pytest.fixture
def pins(tmp_path):
    return core_chaining.PinStore(str(tmp_path / "pins.json"))


class _Model:
    """The two attributes ``ChainingBar`` reads off a settings model."""

    def __init__(self, widgets=None, collect=None):
        self._widgets = dict(widgets or {})
        if collect is not None:
            self.collect = collect


class _Host(QWidget):
    """A module screen with only the seams a test asks for.

    ``AppScreen`` is expensive to build and offers every seam at once, which
    is the opposite of what these tests need: each one removes or breaks a
    single seam and asks what the strip does without it.
    """

    def __init__(self, app_key="measure", *, model=None, run_button=None,
                 worker=None, apply=None):
        super().__init__()
        self.app_key = app_key
        if model is not None:
            self._settings_model = model
        if run_button is not None:
            self._btn_run = run_button
        if worker is not None:
            self._worker = worker
        if apply is not None:
            self.apply_settings_dict = apply


@pytest.fixture
def bar(qtbot, pins):
    """A strip on a bare host — no model, no run button, no worker."""
    host = _Host()
    qtbot.addWidget(host)
    strip = ChainingBar(host, pins=pins)
    qtbot.addWidget(strip)
    return strip


# ---------------------------------------------------------------------------
# Reading a settings widget
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Coming back to the screen
# ---------------------------------------------------------------------------


def test_a_refresh_that_throws_on_show_does_not_reach_qt(qtbot, caplog):
    """An exception out of an event filter is a crash in Qt, not a traceback."""
    def explode():
        raise RuntimeError("registry is locked")

    watched = QWidget()
    qtbot.addWidget(watched)
    watcher = qt_chaining._ShowFilter(explode, watched)
    watched.installEventFilter(watcher)
    with caplog.at_level(logging.ERROR, logger="spacr.qt.chaining"):
        assert watcher.eventFilter(watched, QEvent(QEvent.Show)) is False
    assert "chaining refresh failed on show" in caplog.text


# ---------------------------------------------------------------------------
# Which settings keys the strip is about
# ---------------------------------------------------------------------------


def test_the_bound_keys_are_worked_out_once(qtbot, pins, monkeypatch):
    """Mask's settings map has two hundred entries; a keystroke is not a walk."""
    field = QLineEdit()
    qtbot.addWidget(field)
    host = _Host(model=_Model({"src": field}))
    qtbot.addWidget(host)
    strip = ChainingBar(host, pins=pins)
    qtbot.addWidget(strip)
    assert strip._bound_settings() == ("src",)

    calls = []
    real = core_ports.module_ports
    monkeypatch.setattr(core_ports, "module_ports",
                        lambda key: calls.append(key) or real(key))
    assert strip._bound_settings() == ("src",)
    assert calls == []


# ---------------------------------------------------------------------------
# Following the run
# ---------------------------------------------------------------------------

def test_a_screen_with_no_run_button_is_not_followed(qtbot, pins):
    """A screen with no Run button is a legitimate host for the strip.

    So the strip has to LOOK for the button and stop there -- which is what
    separates "there was nothing to connect" from "it connected to
    something else", both of which raise nothing.
    """
    asked = []

    class NoRunButton(_Host):
        def __getattr__(self, name):      # only reached for missing seams
            asked.append(name)
            raise AttributeError(name)

    host = NoRunButton()
    qtbot.addWidget(host)
    strip = ChainingBar(host, pins=pins)
    qtbot.addWidget(strip)

    asked.clear()                          # the constructor looks around too
    strip._wire_run_button()
    assert asked == ["_btn_run"]


def test_a_run_with_no_project_root_remembers_nothing(bar):
    assert bar._remember_project() == ""


def test_a_project_folder_that_cannot_be_remembered(qtbot, pins, tmp_path,
                                                    monkeypatch, caplog):
    """The recent-source store is where a blank Measure screen looks first."""
    from spacr.qt import prefs

    field = QLineEdit(str(tmp_path))
    qtbot.addWidget(field)
    host = _Host(model=_Model({"src": field}))
    qtbot.addWidget(host)
    strip = ChainingBar(host, pins=pins)
    qtbot.addWidget(strip)

    monkeypatch.setattr(prefs, "push_recent_source",
                        lambda key, root: (_ for _ in ()).throw(
                            RuntimeError("settings are read-only")))
    with caplog.at_level(logging.ERROR, logger="spacr.qt.chaining"):
        assert strip._remember_project() == str(tmp_path)
    assert "could not remember measure's project folder" in caplog.text


# ---------------------------------------------------------------------------
# Seeding, and taking the offered path
# ---------------------------------------------------------------------------

def test_a_seed_the_screen_refuses_is_logged_and_still_offered(qtbot, pins,
                                                               caplog):
    def apply(_values):
        raise RuntimeError("the form is locked")

    host = _Host(apply=apply)
    qtbot.addWidget(host)
    strip = ChainingBar(host, pins=pins)
    qtbot.addWidget(strip)
    with caplog.at_level(logging.ERROR, logger="spacr.qt.chaining"):
        assert strip.adopt({"src": "/plate/a"}) == 0
    assert "could not seed measure" in caplog.text
    assert strip._offered["src"] == "/plate/a", \
        "a seed that failed to apply must still not read back as a user's pin"


# ---------------------------------------------------------------------------
# Continuing to the next module
# ---------------------------------------------------------------------------

def _step(module="classify", seed=None):
    return NextStep(module=module, source="measure", root="/plate",
                    kinds=("crops",), seed=dict(seed or {"src": "/plate"}),
                    readiness=None, artifacts=())


def test_a_screen_outside_a_window_has_no_navigation_host(bar, monkeypatch):
    monkeypatch.setattr(type(bar._screen), "window",
                        lambda self: (_ for _ in ()).throw(
                            RuntimeError("no parent")), raising=False)
    assert bar.host_window() is None
    bar._on_continue(_step())        # nothing to navigate; not a crash


# ---------------------------------------------------------------------------
# Reading the screen's settings
# ---------------------------------------------------------------------------

def test_a_form_that_will_not_collect_falls_back_to_the_bound_fields(qtbot,
                                                                     pins):
    """The half-filled form is exactly when the strip is most useful."""
    field = QLineEdit("/plate/half")
    qtbot.addWidget(field)

    def collect():
        raise ValueError("cell_channel is required")

    host = _Host(model=_Model({"src": field}, collect=collect))
    qtbot.addWidget(host)
    strip = ChainingBar(host, pins=pins)
    qtbot.addWidget(strip)

    assert strip.current_settings() == {"src": "/plate/half"}
    assert strip._collect_ok is False


def test_a_form_that_collects_reports_the_whole_dict(qtbot, pins):
    host = _Host(model=_Model({}, collect=lambda: {"src": "/plate", "a": 1}))
    qtbot.addWidget(host)
    strip = ChainingBar(host, pins=pins)
    qtbot.addWidget(strip)
    assert strip.current_settings() == {"src": "/plate", "a": 1}
    assert strip._collect_ok is True


def test_without_the_recent_source_store_there_is_nowhere_to_look(bar,
                                                                  monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "spacr.qt.prefs", None)
    assert bar.search_roots() == ()


# ---------------------------------------------------------------------------
# The rows
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def test_a_host_missing_a_slot_is_skipped_rather_than_crashed_on(qtbot):
    """``_build_screen`` is called against a stand-in host by the smoke test.

    Skipped has to mean no connection was made: a signal wired to a slot
    the host does not own fails later, on emit, far from here. And with no
    host at all the screen is not even inspected.
    """
    class Recording:
        def __init__(self):
            self.connected = []

        def connect(self, slot):
            self.connected.append(slot)

    class Screen:
        def __init__(self):
            self.error_explain_requested = Recording()
            self.remote_submit_requested = Recording()

    screen = Screen()
    qt_chaining._connect_host(screen, object())      # a host with no slots
    assert screen.error_explain_requested.connected == []
    assert screen.remote_submit_requested.connected == []

    asked = []

    class Watched:
        def __getattr__(self, name):
            asked.append(name)
            raise AttributeError(name)

    qt_chaining._connect_host(Watched(), None)
    assert asked == []


def test_a_connection_the_host_refuses_is_logged_not_raised(caplog):
    class Signal_:
        def connect(self, _slot):
            raise RuntimeError("wrong thread")

    class Screen:
        error_explain_requested = Signal_()
        remote_submit_requested = Signal_()

    class Host:
        def _on_explain_error(self, *a):
            pass

        def _on_remote_submit_requested(self, *a):
            pass

    with caplog.at_level(logging.ERROR, logger="spacr.qt.chaining"):
        qt_chaining._connect_host(Screen(), Host())
    assert "could not connect error_explain_requested" in caplog.text


def test_the_offer_list_is_returned_unfiltered_without_an_app_list(
        monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "spacr.qt.app", None)
    steps = (_step(),)
    assert qt_chaining._only_what_the_gui_offers(steps) is steps


def test_a_step_that_cannot_be_renamed_keeps_the_key_it_had(monkeypatch):
    import dataclasses

    step = _step()
    monkeypatch.setattr(dataclasses, "replace",
                        lambda *a, **k: (_ for _ in ()).throw(
                            TypeError("not a dataclass")))
    assert qt_chaining._renamed(step, "other") is step


def test_registration_survives_a_stylesheet_that_will_not_register(
        monkeypatch, caplog):
    from spacr.qt import theme

    monkeypatch.setattr(theme, "register_widget_qss",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("registry closed")))
    with caplog.at_level(logging.ERROR, logger="spacr.qt.chaining"):
        assert qt_chaining.register() is True
    assert "could not register the chaining strip's stylesheet" in caplog.text
    assert qt_chaining.unregister() > 0
