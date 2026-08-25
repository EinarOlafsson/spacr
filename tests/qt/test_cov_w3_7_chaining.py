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

from PySide6.QtCore import QEvent, QObject, Signal
from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton, QWidget

from spacr import chaining as core_chaining
from spacr import ports as core_ports
from spacr.chaining import ChainedInput, HeldPin, NextStep
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


class _Worker(QObject):
    finished = Signal(bool)


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

def test_a_widget_is_read_through_get_value_first():
    class Editor:
        def get_value(self):
            return ["/plate/a"]

        def text(self):
            raise AssertionError("text() was preferred over get_value()")

    assert qt_chaining._widget_value(Editor()) == ["/plate/a"]


def test_a_plain_line_edit_is_read_through_text(qtbot):
    field = QLineEdit("/plate/b")
    qtbot.addWidget(field)
    assert qt_chaining._widget_value(field) == "/plate/b"


def test_a_getter_that_raises_reads_as_no_value_not_as_a_crash():
    """Half-built editors raise while the user is still typing in them."""
    class Editor:
        def get_value(self):
            raise RuntimeError("half-built")

    class Field:
        def text(self):
            raise RuntimeError("gone")

    assert qt_chaining._widget_value(Editor()) is None
    assert qt_chaining._widget_value(Field()) is None
    assert qt_chaining._widget_value(QObject()) is None
    assert qt_chaining._widget_value(None) is None


# ---------------------------------------------------------------------------
# Coming back to the screen
# ---------------------------------------------------------------------------

def test_showing_the_screen_again_schedules_a_refresh(qtbot, pins):
    """A run on another tab may have produced the thing this strip is about."""
    host = _Host()
    qtbot.addWidget(host)
    strip = ChainingBar(host, pins=pins)
    qtbot.addWidget(strip)
    assert not strip._timer.isActive()

    host.show()
    qtbot.waitExposed(host)
    assert strip._timer.isActive(), \
        "returning to the screen did not schedule a re-read of the registry"
    strip._timer.stop()
    host.hide()


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


def test_a_screen_that_will_not_be_watched_still_gets_its_strip(qtbot, pins,
                                                                caplog):
    host = _Host()
    qtbot.addWidget(host)
    host.installEventFilter = lambda _f: (_ for _ in ()).throw(
        RuntimeError("no event loop"))
    with caplog.at_level(logging.ERROR, logger="spacr.qt.chaining"):
        strip = ChainingBar(host, pins=pins)
    qtbot.addWidget(strip)
    assert "could not watch" in caplog.text
    strip.refresh()          # and it still works


# ---------------------------------------------------------------------------
# Which settings keys the strip is about
# ---------------------------------------------------------------------------

def test_a_module_whose_ports_cannot_be_read_binds_nothing(qtbot, pins,
                                                            monkeypatch):
    host = _Host()
    qtbot.addWidget(host)
    monkeypatch.setattr(core_ports, "module_ports",
                        lambda key: (_ for _ in ()).throw(
                            RuntimeError("bad port table")))
    strip = ChainingBar(host, pins=pins)
    qtbot.addWidget(strip)
    assert strip._bound_settings() == ()
    strip.refresh()          # logged, and the screen it sits on survives
    assert strip._held == {}


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


def test_a_field_whose_edit_signal_refuses_the_connection(qtbot, pins,
                                                          caplog):
    class Refuses:
        def connect(self, _slot):
            raise RuntimeError("wrong thread")

    class Field:
        textEdited = Refuses()

        def get_value(self):
            return ""

    host = _Host(model=_Model({"src": Field()}))
    qtbot.addWidget(host)
    with caplog.at_level(logging.ERROR, logger="spacr.qt.chaining"):
        strip = ChainingBar(host, pins=pins)
    qtbot.addWidget(strip)
    assert "could not watch measure.src for edits" in caplog.text


def test_a_typed_path_schedules_one_refresh_not_one_per_keystroke(qtbot,
                                                                   pins):
    field = QLineEdit()
    qtbot.addWidget(field)
    host = _Host(model=_Model({"src": field}))
    qtbot.addWidget(host)
    strip = ChainingBar(host, pins=pins)
    qtbot.addWidget(strip)

    field.setText("/typed/by/hand")      # not a user edit: no signal
    assert not strip._timer.isActive()
    qtbot.keyClicks(field, "x")
    assert strip._timer.isActive()
    assert strip._timer.isSingleShot()
    strip._timer.stop()


# ---------------------------------------------------------------------------
# Following the run
# ---------------------------------------------------------------------------

def test_a_screen_with_no_run_button_is_not_followed(bar):
    bar._wire_run_button()   # must not raise; there is nothing to connect


def test_a_run_button_that_refuses_the_connection(qtbot, pins, caplog):
    class Refuses:
        def connect(self, _slot):
            raise RuntimeError("wrong thread")

    class Button:
        clicked = Refuses()

    host = _Host(run_button=Button())
    qtbot.addWidget(host)
    with caplog.at_level(logging.ERROR, logger="spacr.qt.chaining"):
        strip = ChainingBar(host, pins=pins)
    qtbot.addWidget(strip)
    assert "could not follow the Run button on measure" in caplog.text


def test_pressing_run_without_a_worker_is_not_a_crash(qtbot, pins):
    button = QPushButton()
    qtbot.addWidget(button)
    host = _Host(run_button=button)
    qtbot.addWidget(host)
    strip = ChainingBar(host, pins=pins)
    qtbot.addWidget(strip)
    button.click()           # the run never started; nothing to follow


def test_the_finished_run_is_followed_and_offers_the_next_step(qtbot, pins,
                                                               monkeypatch):
    button = QPushButton()
    qtbot.addWidget(button)
    worker = _Worker()
    host = _Host(run_button=button, worker=worker)
    qtbot.addWidget(host)
    strip = ChainingBar(host, pins=pins)
    qtbot.addWidget(strip)

    seen = []
    monkeypatch.setattr(type(strip), "refresh",
                        lambda self, **kwargs: seen.append(kwargs))
    button.click()
    worker.finished.emit(True)
    assert {"finished": True} in seen


def test_a_worker_that_cannot_be_followed_does_not_stop_the_run(qtbot, pins,
                                                                caplog):
    class Worker:
        class finished:
            @staticmethod
            def connect(_slot):
                raise RuntimeError("already gone")

    button = QPushButton()
    qtbot.addWidget(button)
    host = _Host(run_button=button, worker=Worker())
    qtbot.addWidget(host)
    strip = ChainingBar(host, pins=pins)
    qtbot.addWidget(strip)
    with caplog.at_level(logging.ERROR, logger="spacr.qt.chaining"):
        button.click()
    assert "could not follow the measure run to its end" in caplog.text


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


def test_a_pin_whose_upstream_has_not_moved_is_left_where_it_is(bar):
    """"Use it" is about the pins that differ; the rest are already right."""
    bar._held = {"src": HeldPin("src", "/mine", offered=None)}
    bar._pins.pin("measure", "src", "/mine")
    bar._on_use_offered()
    assert bar._pins.pinned("measure", "src") == "/mine"


def test_taking_the_offered_path_drops_the_pin_and_fills_the_field(qtbot,
                                                                   pins):
    applied = {}
    host = _Host(apply=lambda values: applied.update(values) or len(values))
    qtbot.addWidget(host)
    strip = ChainingBar(host, pins=pins)
    qtbot.addWidget(strip)
    pins.pin("measure", "src", "/old/plate")
    strip._held = {"src": _moved()}

    strip._on_use_offered()
    assert applied == {"src": "/new/plate"}
    assert pins.pinned("measure", "src") is None
    assert strip._offered["src"] == "/new/plate"


def test_an_offered_path_the_screen_will_not_take_still_drops_the_pin(
        qtbot, pins, caplog):
    host = _Host(apply=lambda values: (_ for _ in ()).throw(
        RuntimeError("locked")))
    qtbot.addWidget(host)
    strip = ChainingBar(host, pins=pins)
    qtbot.addWidget(strip)
    pins.pin("measure", "src", "/old/plate")
    strip._held = {"src": _moved()}
    with caplog.at_level(logging.ERROR, logger="spacr.qt.chaining"):
        strip._on_use_offered()
    assert "could not apply the offered path for measure" in caplog.text
    assert pins.pinned("measure", "src") is None


# ---------------------------------------------------------------------------
# Continuing to the next module
# ---------------------------------------------------------------------------

def _moved(value="/old/plate", offered="/new/plate"):
    """A pin whose upstream has since written somewhere else.

    ``HeldPin.differs`` is only true when there is a resolved upstream to
    differ FROM, so the artifact behind the offer is built rather than
    implied.
    """
    from spacr.artifacts import Artifact

    artifact = Artifact(
        artifact_id="0" * 16, project="/plate", kind="merged-arrays",
        role="merged", path=offered, module="mask", run_id="run",
        settings_hash="h", spacr_version="0", created_ns=0,
        created_utc="1970-01-01T00:00:00Z", fingerprint="f",
        fingerprint_method="none", size_bytes=0, n_files=1,
        status="complete")
    chained = ChainedInput(module="measure", setting="src", role="merged",
                           kind="merged-arrays", value=offered,
                           artifact=artifact, producer="mask", root="/plate")
    return HeldPin("src", value, offered=offered, chained=chained)


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


def test_a_host_that_cannot_navigate_is_not_navigated(bar, monkeypatch):
    monkeypatch.setattr(type(bar), "host_window", lambda self: object())
    bar._on_continue(_step())


def test_a_navigation_that_fails_does_not_seed_anything(bar, monkeypatch):
    class Window:
        _screens = {"classify": object()}

        def _on_nav_selected(self, module):
            raise RuntimeError("no such page")

    monkeypatch.setattr(type(bar), "host_window", lambda self: Window())
    bar._on_continue(_step())        # logged, and no seeding attempted


def test_continuing_to_a_screen_that_is_not_built_yet(bar, monkeypatch):
    class Window:
        _screens: dict = {}

        def _on_nav_selected(self, module):
            self.opened = module

    window = Window()
    monkeypatch.setattr(type(bar), "host_window", lambda self: window)
    bar._on_continue(_step())
    assert window.opened == "classify"


def test_continuing_seeds_the_successors_own_strip_when_it_has_one(bar,
                                                                   monkeypatch):
    adopted = {}

    class Target:
        _chaining_bar = type("B", (), {"adopt": lambda self, seed:
                                       adopted.update(seed)})()

        def apply_settings_dict(self, values):
            raise AssertionError("seeded around the successor's own strip")

    class Window:
        _screens = {"classify": Target()}

        def _on_nav_selected(self, module):
            pass

    monkeypatch.setattr(type(bar), "host_window", lambda self: Window())
    bar._on_continue(_step(seed={"src": "/plate/x"}))
    assert adopted == {"src": "/plate/x"}


def test_a_successor_without_a_strip_is_seeded_directly(bar, monkeypatch):
    seeded = {}

    class Target:
        def apply_settings_dict(self, values):
            seeded.update(values)
            return len(values)

    class Window:
        _screens = {"classify": Target()}

        def _on_nav_selected(self, module):
            pass

    monkeypatch.setattr(type(bar), "host_window", lambda self: Window())
    bar._on_continue(_step(seed={"src": "/plate/y"}))
    assert seeded == {"src": "/plate/y"}


def test_a_successor_that_refuses_the_seed_still_opens(bar, monkeypatch,
                                                       caplog):
    class Target:
        def apply_settings_dict(self, values):
            raise RuntimeError("locked")

    class Window:
        _screens = {"classify": Target()}

        def _on_nav_selected(self, module):
            self.opened = module

    window = Window()
    monkeypatch.setattr(type(bar), "host_window", lambda self: window)
    with caplog.at_level(logging.ERROR, logger="spacr.qt.chaining"):
        bar._on_continue(_step())
    assert window.opened == "classify"
    assert "could not seed classify" in caplog.text


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


def test_a_module_with_no_upstream_still_searches_its_own_folder(bar,
                                                                 monkeypatch):
    from spacr.qt import prefs

    monkeypatch.setattr(core_ports, "upstream_modules",
                        lambda key: (_ for _ in ()).throw(
                            RuntimeError("no graph")))
    monkeypatch.setattr(prefs, "get_last_source",
                        lambda key: "/plate/last" if key == "measure" else "")
    monkeypatch.setattr(prefs, "get_recent_sources", lambda key, limit=4: [])
    assert bar.search_roots() == ("/plate/last",)


# ---------------------------------------------------------------------------
# The rows
# ---------------------------------------------------------------------------

def test_a_refresh_that_throws_is_logged_and_the_screen_survives(bar,
                                                                 monkeypatch,
                                                                 caplog):
    monkeypatch.setattr(type(bar), "_refresh",
                        lambda self, **kwargs: (_ for _ in ()).throw(
                            RuntimeError("registry is corrupt")))
    with caplog.at_level(logging.ERROR, logger="spacr.qt.chaining"):
        bar.refresh()
    assert "could not refresh the chaining strip for measure" in caplog.text


def test_a_successor_search_that_fails_offers_nothing(bar, monkeypatch,
                                                      caplog):
    monkeypatch.setattr(core_chaining, "next_steps",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("registry is corrupt")))
    with caplog.at_level(logging.ERROR, logger="spacr.qt.chaining"):
        bar._draw_next({"src": "/plate"}, finished=True)
    assert "could not work out what comes after measure" in caplog.text
    assert bar._next_row.isHidden()
    assert bar._last_steps == ()


def test_a_finished_run_with_no_successor_hides_the_row(bar, monkeypatch):
    monkeypatch.setattr(core_chaining, "next_steps", lambda *a, **k: ())
    bar._draw_next({"src": "/plate"}, finished=True)
    assert bar._next_row.isHidden()


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def test_a_screen_without_the_layout_seams_gets_no_strip(qtbot, pins):
    host = _Host()
    qtbot.addWidget(host)
    assert qt_chaining.install_chaining(host, pins=pins) is None

    host._runtime_wrap = QWidget()
    host._actions_row = QWidget()
    qtbot.addWidget(host._runtime_wrap)
    qtbot.addWidget(host._actions_row)
    assert host._runtime_wrap.layout() is None
    assert qt_chaining.install_chaining(host, pins=pins) is None


def test_a_host_missing_a_slot_is_skipped_rather_than_crashed_on(qtbot):
    """``_build_screen`` is called against a stand-in host by the smoke test."""
    class Screen:
        error_explain_requested = None
        remote_submit_requested = None

    qt_chaining._connect_host(Screen(), object())
    qt_chaining._connect_host(Screen(), None)


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
