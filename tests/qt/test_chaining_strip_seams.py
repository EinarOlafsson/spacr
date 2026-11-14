"""Table-driven tests of the chaining strip ``spacr.qt.chaining.ChainingBar``.

The strip is an aid that must never take a module screen down with it.  One
general test per behaviour, each walking a case table: the screen seams it
cannot watch or follow, the single debounced refresh, following a run to its
end, "Use it" on a moved pin, Continue (navigate first, seed second), the
successor row, a chained path on a screen that cannot be written to, the
folders it searches, and installing the strip with its surfaces swept.  The
strip is built on a stand-in host offering only the seams each case asks for.
"""
from __future__ import annotations

import logging
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtWidgets import (QApplication, QLineEdit, QPushButton,  # noqa: E402
                               QSpacerItem, QWidget)

from spacr import chaining as core_chaining  # noqa: E402
from spacr import ports as core_ports  # noqa: E402
from spacr.chaining import (ChainedInput, HeldPin, NextStep,  # noqa: E402
                            Resolution)
from spacr.qt import chaining as qt_chaining  # noqa: E402
from spacr.qt.chaining import ChainingBar  # noqa: E402

pytestmark = pytest.mark.qt


def _settle(bar, *, timeout: float = 15.0) -> None:
    """Wait until the asynchronous refresh and its coalesced follow-up have landed."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if not getattr(bar, "_resolving", False) and not getattr(
                bar, "_resolve_again", False):
            return
    raise AssertionError("the chaining strip never finished resolving")


@pytest.fixture(autouse=True)
def _own_pins(monkeypatch, tmp_path):
    """Never read or write the developer's real pin file."""
    monkeypatch.setenv(core_chaining.PIN_STATE_ENV, str(tmp_path / "pins.json"))
    core_chaining.pin_store(refresh=True)
    yield
    core_chaining.pin_store(refresh=True)


@pytest.fixture
def pins(tmp_path):
    """A private pin store."""
    return core_chaining.PinStore(str(tmp_path / "pins.json"))


class _Model:
    """The two attributes ``ChainingBar`` reads off a settings model."""

    def __init__(self, widgets=None, collect=None):
        self._widgets = dict(widgets or {})
        if collect is not None:
            self.collect = collect


class _Host(QWidget):
    """A module screen with only the seams a test asks for."""

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


class _Refuses:
    """A signal whose ``connect`` raises."""

    def connect(self, _slot):
        raise RuntimeError("wrong thread")


@pytest.fixture
def make_bar(qtbot, pins):
    """Build a strip on a stand-in host with the given seams."""
    def build(**seams):
        host = _Host(**seams)
        qtbot.addWidget(host)
        strip = ChainingBar(host, pins=pins)
        qtbot.addWidget(strip)
        return strip
    return build


def _errors(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == "spacr.qt.chaining" and r.levelno >= logging.ERROR]


def _moved(value="/old/plate", offered="/new/plate"):
    """A pin whose upstream has since written somewhere else."""
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


# ---------------------------------------------------------------------------
# Construction seams
# ---------------------------------------------------------------------------

def test_the_strip_is_built_whatever_seam_its_screen_refuses(
        subtests, qtbot, pins, monkeypatch, caplog):
    """An unwatchable screen, unreadable ports, a refusing edit signal or Run button: logged, strip built."""
    from spacr.qt import gil_priority

    class RefusingField:
        textEdited = _Refuses()

        def get_value(self):
            return ""

    class RefusingButton:
        clicked = _Refuses()

    def no_event_loop(*_args):
        raise RuntimeError("no event loop")

    def bad_ports(key):
        raise RuntimeError("bad port table")

    cases = [
        # (id, patch, seams, errors, bound)
        ("a screen that will not be watched",
         (gil_priority, "_watch_object_events", no_event_loop), {},
         ["could not watch measure for show events"], ()),
        ("a module whose ports cannot be read binds nothing",
         (core_ports, "module_ports", bad_ports),
         {"model": _Model({"src": QLineEdit()})}, [], ()),
        ("a field whose edit signal refuses", None,
         {"model": _Model({"src": RefusingField()})},
         ["could not watch measure.src for edits"], ("src",)),
        ("a run button that refuses", None, {"run_button": RefusingButton()},
         ["could not follow the Run button on measure"], ()),
    ]
    for case_id, patch, seams, errors, bound in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as m:
            if patch is not None:
                m.setattr(*patch)
            caplog.clear()
            with caplog.at_level(logging.ERROR, logger="spacr.qt.chaining"):
                host = _Host(**seams)
                qtbot.addWidget(host)
                strip = ChainingBar(host, pins=pins)
                qtbot.addWidget(strip)
            assert _errors(caplog) == errors
            assert strip._bound_settings() == bound
            strip.refresh()
            assert strip._held == {}
            assert strip._btn_use.text() == "Use it"


def test_showing_the_screen_or_typing_a_path_schedules_one_refresh(
        subtests, qtbot, make_bar):
    """Show and a user keystroke start the single-shot debounce timer; a programmatic edit does not."""
    with subtests.test(msg="showing the screen again"):
        strip = make_bar()
        assert strip._timer.isActive() is False
        strip._screen.show()
        qtbot.waitExposed(strip._screen)
        assert strip._timer.isActive() is True
        strip._timer.stop()
        strip._screen.hide()

    with subtests.test(msg="typing a path"):
        field = QLineEdit()
        qtbot.addWidget(field)
        strip = make_bar(model=_Model({"src": field}))
        field.setText("/typed/by/hand")
        assert strip._timer.isActive() is False
        qtbot.keyClicks(field, "x")
        assert strip._timer.isActive() is True
        assert strip._timer.isSingleShot() is True
        assert strip._timer.interval() == qt_chaining.REFRESH_DELAY_MS
        strip._timer.stop()


def test_pressing_run_follows_the_worker_to_its_end(subtests, qtbot, make_bar,
                                                    monkeypatch, caplog):
    """The click is captured; a worker's finish refreshes with its outcome; none or a refusing one does not."""
    class GoneWorker:
        class finished:
            @staticmethod
            def connect(_slot):
                raise RuntimeError("already gone")

    cases = [
        # (id, worker at click, emitted, refreshes, errors)
        ("no worker yet: the click is followed, the run is not", None,
         [True], [], []),
        ("a run that succeeds offers the next step", "worker", [True],
         [{"finished": True}], []),
        ("a run that fails refreshes without offering", "worker", [False],
         [{"finished": False}], []),
        ("a worker that cannot be followed", "gone", [], [],
         ["could not follow the measure run to its end"]),
    ]
    for case_id, kind, emitted, refreshes, errors in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as m:
            button = QPushButton()
            qtbot.addWidget(button)
            worker = _Worker()
            seams = {"run_button": button}
            if kind == "worker":
                seams["worker"] = worker
            elif kind == "gone":
                seams["worker"] = GoneWorker()
            strip = make_bar(**seams)
            captured, seen = [], []
            m.setattr(type(strip), "_capture_edits",
                      lambda self: captured.append(True))
            m.setattr(type(strip), "refresh",
                      lambda self, **kwargs: seen.append(kwargs))
            caplog.clear()
            with caplog.at_level(logging.ERROR, logger="spacr.qt.chaining"):
                button.click()
            if kind is None:
                strip._screen._worker = worker
            for ok in emitted:
                worker.finished.emit(ok)
            assert captured == [True]
            assert seen == refreshes
            assert _errors(caplog) == errors


def test_use_it_drops_only_moved_pins_and_offers_their_new_path(
        subtests, make_bar, pins, caplog):
    """Moved pins are dropped and their new path applied; unmoved pins stay; a refusing screen is logged."""
    def refuse(values):
        raise RuntimeError("locked")

    cases = [
        # (id, apply, held, pinned before, applied, src pin, offered, errors)
        ("a pin whose upstream has not moved is left", "record",
         {"src": HeldPin("src", "/mine", offered=None)}, "/mine", [],
         "/mine", {}, []),
        ("taking the offer drops the pin and fills the field", "record",
         {"src": _moved()}, "/old/plate", [{"src": "/new/plate"}], None,
         {"src": "/new/plate"}, []),
        ("a screen that will not take it still drops the pin", refuse,
         {"src": _moved()}, "/old/plate", [], None, {"src": "/new/plate"},
         ["could not apply the offered path for measure"]),
    ]
    for case_id, apply, held, before, applied, after, offered, errors in cases:
        with subtests.test(msg=case_id):
            seen = []
            strip = make_bar(apply=(lambda values: seen.append(dict(values))
                                    or len(values))
                             if apply == "record" else apply)
            pins.pin("measure", "src", before)
            strip._held = dict(held)
            caplog.clear()
            with caplog.at_level(logging.ERROR, logger="spacr.qt.chaining"):
                strip._on_use_offered()
            assert seen == applied
            _settle(strip)
            assert pins.pinned("measure", "src") == after
            assert strip._offered == offered
            assert _errors(caplog) == errors


def test_continue_navigates_first_and_seeds_the_successor_second(
        subtests, make_bar, monkeypatch, caplog):
    """No route stops at once, a failed route seeds nothing, else the successor opens and is seeded."""
    log = []

    class Watched:
        def __init__(self, name):
            self._name = name

        def __getattr__(self, attr):
            log.append((self._name, attr))
            raise AttributeError(attr)

    def window(screens, *, fails=False):
        class Window:
            _screens = screens
            opened = None

            def _on_nav_selected(self, module):
                if fails:
                    raise RuntimeError("no such page")
                self.opened = module

        return Window()

    class OwnStrip:
        def __init__(self):
            self.adopted = []
            self._chaining_bar = type("B", (), {
                "adopt": lambda bar, seed: self.adopted.append(dict(seed))})()

        def apply_settings_dict(self, values):
            raise AssertionError("seeded around the successor's own strip")

    class Direct:
        def __init__(self, fails=False):
            self.seeded = []
            self._fails = fails

        def apply_settings_dict(self, values):
            if self._fails:
                raise RuntimeError("locked")
            self.seeded.append(dict(values))
            return len(values)

    own, direct, refusing = OwnStrip(), Direct(), Direct(fails=True)
    cases = [
        # (id, window, opened, touched, errors, check)
        ("a host that cannot navigate is not navigated", Watched("window"),
         None, [("window", "_on_nav_selected")], [], None),
        ("a navigation that fails seeds nothing",
         window({"classify": Watched("target")}, fails=True), None, [],
         ["could not open classify"], None),
        ("a successor not built yet is only opened", window({}), "classify",
         [], [], None),
        ("a successor with its own strip adopts the seed",
         window({"classify": own}), "classify", [], [],
         lambda: own.adopted == [{"src": "/plate/x"}]),
        ("a successor without a strip is seeded directly",
         window({"classify": direct}), "classify", [], [],
         lambda: direct.seeded == [{"src": "/plate/x"}]),
        ("a successor that refuses the seed still opens",
         window({"classify": refusing}), "classify", [],
         ["could not seed classify"], None),
    ]
    strip = make_bar()
    for case_id, host_window, opened, touched, errors, check in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as m:
            log.clear()
            caplog.clear()
            m.setattr(type(strip), "host_window",
                      lambda self, w=host_window: w)
            with caplog.at_level(logging.ERROR, logger="spacr.qt.chaining"):
                strip._on_continue(_step(seed={"src": "/plate/x"}))
            assert log == touched
            if opened is not None:
                assert host_window.opened == opened
            assert _errors(caplog) == errors
            if check is not None:
                assert check()


def test_the_successor_row_is_rebuilt_from_scratch_or_hidden(
        subtests, make_bar, monkeypatch, caplog):
    """A failing or empty successor search hides the row; stale items between its ends are cleared."""
    def corrupt(*_a, **_k):
        raise RuntimeError("registry is corrupt")

    with subtests.test(msg="a successor search that fails offers nothing"), \
            monkeypatch.context() as m:
        strip = make_bar()
        m.setattr(core_chaining, "next_steps", corrupt)
        strip._last_steps = (_step(),)
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="spacr.qt.chaining"):
            strip._draw_next({"src": "/plate"}, finished=True)
        assert _errors(caplog) == [
            "could not work out what comes after measure"]
        assert (strip._next_row.isHidden(), strip._last_steps) == (True, ())

    with subtests.test(msg="a finished run with no successor hides the row"), \
            monkeypatch.context() as m:
        strip = make_bar()
        m.setattr(core_chaining, "next_steps", lambda *a, **k: ())
        strip._next_row.show()
        strip._draw_next({"src": "/plate"}, finished=True)
        assert (strip._next_row.isHidden(), strip._last_steps) == (True, ())

    with subtests.test(msg="stale buttons and spacers are cleared, the label stays"):
        strip = make_bar()
        stale_button = QPushButton("Continue to Classify")
        strip._next_layout.addWidget(stale_button)
        strip._next_layout.addItem(QSpacerItem(1, 1))
        strip._next_row.show()
        strip._last_steps = (_step(),)
        assert strip._next_layout.count() == 3
        strip._draw_next({"src": "/plate"}, finished=False)
        assert strip._next_layout.count() == 1
        assert strip._next_layout.itemAt(0).widget() is strip._next_label
        assert stale_button.parent() is None
        assert (strip._next_row.isHidden(), strip._last_steps) == (True, ())

    with subtests.test(msg="a refresh that throws is logged"), \
            monkeypatch.context() as m:
        strip = make_bar()
        m.setattr(type(strip), "_refresh",
                  lambda self, **kwargs: corrupt())
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="spacr.qt.chaining"):
            strip.refresh()
            _settle(strip)
        assert _errors(caplog) == [
            "could not refresh the chaining strip for measure"]


def test_a_chained_path_is_recorded_whether_or_not_the_field_can_be_written(
        subtests, qtbot, make_bar, monkeypatch):
    """The registry's answer fills an empty field through the seam, and is remembered as offered either way."""
    monkeypatch.setattr(
        core_chaining, "resolve_settings",
        lambda *a, **k: Resolution(module="measure",
                                   settings={"src": "/plate/from-the-registry"},
                                   filled={}, held={}, inputs=()))
    for case_id, writable, text in (
            ("a screen with the seam is filled", True,
             "/plate/from-the-registry"),
            ("a screen without the seam is left blank", False, "")):
        with subtests.test(msg=case_id):
            field = QLineEdit("")
            qtbot.addWidget(field)
            seams = {"model": _Model({"src": field})}
            if writable:
                seams["apply"] = lambda values: field.setText(values["src"])
            strip = make_bar(**seams)
            assert strip._bound_settings() == ("src",)
            strip.refresh()
            _settle(strip)
            assert field.text() == text
            assert strip._offered == {"src": "/plate/from-the-registry"}
            assert strip._seen == {"src": "/plate/from-the-registry"}


def test_the_search_roots_are_own_then_upstream_folders_that_exist(
        subtests, make_bar, monkeypatch):
    """Own folder first, upstream folders next, duplicates and unknown folders dropped."""
    from spacr.qt import path_probe, prefs

    def no_graph(key):
        raise RuntimeError("no graph")

    last = {"measure": "/plate/last", "mask": "/plate/mask"}
    recent = {"measure": ["/plate/last", "/plate/gone"],
              "mask": ["/plate/older"]}
    for path, isdir in (("/plate/last", True), ("/plate/mask", True),
                        ("/plate/older", True), ("/plate/gone", False)):
        path_probe.prime(path, isdir, want_dir=True)
    cases = [
        ("no upstream graph still searches its own folder", no_graph,
         ("/plate/last",)),
        ("upstream folders follow, in order", lambda key: ["mask"],
         ("/plate/last", "/plate/mask", "/plate/older")),
    ]
    strip = make_bar()
    for case_id, upstream, expected in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as m:
            m.setattr(core_ports, "upstream_modules", upstream)
            m.setattr(prefs, "get_last_source", lambda key: last.get(key, ""))
            m.setattr(prefs, "get_recent_sources",
                      lambda key, limit=4: list(recent.get(key, []))[:limit])
            assert strip.search_roots() == expected


# ---------------------------------------------------------------------------
# install_chaining
# ---------------------------------------------------------------------------

def test_installing_the_strip_sweeps_its_scaffolding_but_not_its_surfaces(
        subtests, qtbot, pins, monkeypatch):
    """Missing seams give no strip; on a real screen anonymous rows are swept, named parts kept, once."""
    from spacr.qt import theme
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.theme import TRANSPARENT_PROPERTY

    def tagged(widget):
        return bool(widget.property(TRANSPARENT_PROPERTY))

    def regression_screen():
        screen = AppScreen("regression")
        qtbot.addWidget(screen)
        screen._clear_page_surfaces()
        return screen

    with subtests.test(msg="a screen without the layout seams gets no strip"):
        host = _Host()
        qtbot.addWidget(host)
        assert qt_chaining.install_chaining(host, pins=pins) is None
        host._runtime_wrap = QWidget()
        host._actions_row = QWidget()
        qtbot.addWidget(host._runtime_wrap)
        qtbot.addWidget(host._actions_row)
        assert qt_chaining.install_chaining(host, pins=pins) is None
        assert qt_chaining.chaining_bar(host) is None

    screen = regression_screen()
    bar = qt_chaining.install_chaining(screen)
    with subtests.test(msg="the anonymous rows are swept"):
        assert type(bar) is ChainingBar
        assert tagged(bar._pinned_row) is True
        assert tagged(bar._next_row) is True
        missed = [w for w in bar.findChildren(QWidget)
                  if type(w) is QWidget and not w.objectName()
                  and not tagged(w)]
        assert missed == []

    with subtests.test(msg="the strip and its named labels keep their surface"):
        assert tagged(bar) is False
        for label, name in ((bar._pinned, "ChainingPinned"),
                            (bar._source, "ChainingSource"),
                            (bar._stale, "ChainingStale"),
                            (bar._fix, "ChainingFix")):
            assert label.objectName() == name
            assert tagged(label) is False

    with subtests.test(msg="installing twice returns the same strip"):
        assert qt_chaining.install_chaining(screen) is bar
        assert screen.findChildren(ChainingBar) == [bar]
        assert qt_chaining.chaining_bar(screen) is bar

    with subtests.test(msg="a sweep that fails still leaves a working strip"), \
            monkeypatch.context() as m:
        def explode(*_a, **_k):
            raise RuntimeError("no style on this platform")

        other = regression_screen()
        m.setattr(theme, "clear_container_surfaces", explode)
        swept = qt_chaining.install_chaining(other)
        assert type(swept) is ChainingBar
        assert swept._btn_use.text() == "Use it"
        assert qt_chaining.chaining_bar(other) is swept
