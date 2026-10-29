"""The data-library warm and a module screen's first build never import together.

A Windows cold launch crashed with 0xC0000005 (1 in 11, release candidate
c21d42a90) while the ``spacr-data-libraries`` thread imported
``skimage.morphology`` and the GUI thread built Annotate, whose first build
imports scikit-image and PIL's Qt bridge. A traced Linux sweep showed the same
shape: Annotate's first build loading ``skimage._shared.geometry`` and
cryptography's native modules while the warm loaded pandas'. Each warm step
now holds ``_DATA_LIBRARY_TURN``; a first open waits for it with the event
loop breathing, and a prewarm step leaves it to its next slice.

These tests use small modules that record when their bodies run, so the
timing is theirs to control.
"""
from __future__ import annotations

import sys
import textwrap
import threading
import time
import uuid

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QObject  # noqa: E402

from spacr.qt import app as appmod  # noqa: E402

WARM_STEP_S = 0.2
OPEN_S = 0.06


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    """Five warm modules and eight screen modules that log their bodies."""
    tag = uuid.uuid4().hex[:8]
    log_name = f"turnlog_{tag}"
    (tmp_path / f"{log_name}.py").write_text(
        "import threading\nLOCK = threading.Lock()\nEVENTS = []\n", encoding="utf-8")
    body = textwrap.dedent(f"""
        import threading, time, {log_name} as log
        start = time.perf_counter()
        time.sleep({{pause}})
        with log.LOCK:
            log.EVENTS.append(({{kind!r}}, threading.current_thread().name,
                               start, time.perf_counter()))
        """)
    warm = tuple(f"warm_{tag}_{i}" for i in range(5))
    screens = tuple(f"screen_{tag}_{i}" for i in range(8))
    for name in warm:
        (tmp_path / f"{name}.py").write_text(
            body.format(pause=WARM_STEP_S, kind="warm"), encoding="utf-8")
    for name in screens:
        (tmp_path / f"{name}.py").write_text(
            body.format(pause=OPEN_S, kind="open"), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(appmod, "_DATA_LIBRARIES", warm)
    monkeypatch.setattr(appmod, "_DATA_LIBRARIES_STARTED", False)
    monkeypatch.setattr(appmod, "_BACKGROUND_WARMS_STOPPING", threading.Event())
    monkeypatch.setattr(appmod, "_freeze_what_survived", lambda: None)
    import spacr._segmentation_backends as backends
    from spacr.qt.widgets import organism_diagram

    monkeypatch.setattr(backends, "_warm_backend_lookups", lambda: None)
    monkeypatch.setattr(organism_diagram, "_warm_the_artwork", lambda: 0)
    log = __import__(log_name)
    yield {"warm": warm, "screens": screens, "log": log}
    for name in list(sys.modules):
        if tag in name:
            sys.modules.pop(name, None)


class _Window(QObject):
    """The parts of ``MainWindow`` a navigation touches, with a fake open."""

    def __init__(self):
        super().__init__()
        self._screens = {}
        self._opening_a_screen = False
        self.deferred = 0
        self.first_click = {}
        self.waits = []

    _on_nav_selected = appmod.MainWindow._on_nav_selected
    _wait_for_the_data_library_turn = appmod.MainWindow._wait_for_the_data_library_turn
    _run_one_prewarm_step = appmod.MainWindow._run_one_prewarm_step

    def _open_a_module_screen(self, key, interaction_started):
        self.waits.append(time.perf_counter() - self.first_click[key])
        __import__(key)
        self._screens[key] = object()


def _overlaps(events):
    warm = [(s, e) for kind, _t, s, e in events if kind == "warm"]
    opens = [(s, e) for kind, _t, s, e in events if kind == "open"]
    return [(w, o) for w in warm for o in opens if w[0] < o[1] and o[0] < w[1]]


class _CountingEvent(threading.Event):
    """``_A_FIRST_OPEN_WAITS``, counting how often a first open had to wait."""

    def __init__(self):
        super().__init__()
        self.times_set = 0

    def set(self):
        self.times_set += 1
        super().set()


def _navigate_while_the_warm_runs(qapp, recorded, window, monkeypatch):
    """Open every screen, one click every 20 ms, while the warm imports."""
    waits = _CountingEvent()
    monkeypatch.setattr(appmod, "_A_FIRST_OPEN_WAITS", waits)
    thread = appmod._import_the_data_libraries_off_the_gui_thread()
    assert thread is not None
    for key in recorded["screens"]:
        window.first_click[key] = time.perf_counter()
        window._on_nav_selected(key)
        assert key in window._screens, "the open returned before the screen was built"
        QCoreApplication.processEvents()
        time.sleep(0.02)
    thread.join(30)
    assert not thread.is_alive()
    window.deferred = waits.times_set
    return list(recorded["log"].EVENTS)


def test_a_first_open_never_runs_beside_a_warm_import(qapp, recorded, monkeypatch):
    monkeypatch.setattr(appmod._timing, "interval_started", lambda *a, **k: None)
    window = _Window()
    events = _navigate_while_the_warm_runs(qapp, recorded, window, monkeypatch)

    kinds = [kind for kind, *_ in events]
    assert kinds.count("warm") == len(recorded["warm"])
    assert kinds.count("open") == len(recorded["screens"])
    assert window.deferred > 0, "no open met a warm step; this test proves nothing"
    assert _overlaps(events) == []
    assert not appmod._DATA_LIBRARY_TURN.locked()


def test_a_waiting_first_open_goes_before_the_next_warm_step(qapp, recorded, monkeypatch):
    """No starvation: a click waits for one warm step, not for the whole warm."""
    monkeypatch.setattr(appmod._timing, "interval_started", lambda *a, **k: None)
    window = _Window()
    events = _navigate_while_the_warm_runs(qapp, recorded, window, monkeypatch)
    assert window.deferred > 0, "no open met a warm step; this test proves nothing"
    assert max(window.waits) < 2.5 * WARM_STEP_S, window.waits
    warm_ends = [e for kind, _t, _s, e in events if kind == "warm"]
    open_starts = [s for kind, _t, s, _e in events if kind == "open"]
    assert min(open_starts) < max(warm_ends), "every open waited for the whole warm"


def test_without_the_turn_the_two_do_overlap(qapp, recorded, monkeypatch):
    """The control: with the turn always free, opens land inside warm imports."""
    import contextlib

    monkeypatch.setattr(appmod._timing, "interval_started", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_the_data_library_turn",
                        lambda: contextlib.nullcontext(True))
    window = _Window()
    events = _navigate_while_the_warm_runs(qapp, recorded, window, monkeypatch)
    assert _overlaps(events), "the control never overlapped; the timing proves nothing"


def test_a_prewarm_step_waits_for_the_warm_step_in_flight(qapp, monkeypatch):
    ran = []

    def steps():
        ran.append("step")
        yield

    class Prewarm:
        _steps = steps()
        _key = "x"
        built = []

    window = _Window()
    assert appmod._DATA_LIBRARY_TURN.acquire(timeout=5)
    try:
        window._run_one_prewarm_step(Prewarm)
        assert ran == []
        assert window._opening_a_screen is False
    finally:
        appmod._DATA_LIBRARY_TURN.release()
    window._run_one_prewarm_step(Prewarm)
    assert ran == ["step"]
    assert not appmod._DATA_LIBRARY_TURN.locked()


def test_a_warm_waiting_for_its_turn_stops_when_the_application_quits(monkeypatch):
    monkeypatch.setattr(appmod, "_BACKGROUND_WARMS_STOPPING", threading.Event())
    assert appmod._DATA_LIBRARY_TURN.acquire(timeout=5)
    answers = []
    try:
        waiter = threading.Thread(
            target=lambda: answers.append(
                appmod._the_data_library_turn().__enter__()))
        waiter.start()
        time.sleep(0.15)
        appmod._BACKGROUND_WARMS_STOPPING.set()
        waiter.join(5)
    finally:
        appmod._DATA_LIBRARY_TURN.release()
    assert answers == [False]
    assert not appmod._DATA_LIBRARY_TURN.locked()


def test_reopening_a_built_screen_does_not_wait(qapp, monkeypatch):
    monkeypatch.setattr(appmod._timing, "interval_started", lambda *a, **k: None)
    window = _Window()
    window._screens["built"] = object()
    opened = []
    window._open_a_module_screen = lambda key, started: opened.append(key)
    assert appmod._DATA_LIBRARY_TURN.acquire(timeout=5)
    try:
        window._on_nav_selected("built")
    finally:
        appmod._DATA_LIBRARY_TURN.release()
    assert opened == ["built"]


def test_a_first_open_that_cannot_get_the_turn_still_opens(qapp, monkeypatch):
    """A warm stuck in one step delays a first open by the bound, not for ever."""
    monkeypatch.setattr(appmod._timing, "interval_started", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_DATA_LIBRARY_TURN_WAIT_S", 0.2)
    window = _Window()
    opened = []
    window._open_a_module_screen = lambda key, started: opened.append(key)
    assert appmod._DATA_LIBRARY_TURN.acquire(timeout=5)
    try:
        started = time.monotonic()
        window._on_nav_selected("stuck")
        assert time.monotonic() - started < 2
    finally:
        appmod._DATA_LIBRARY_TURN.release()
    assert opened == ["stuck"]
    assert window._opening_a_screen is False
    assert not appmod._A_FIRST_OPEN_WAITS.is_set()


def test_quit_during_first_open_priority_releases_the_warm_without_taking_the_turn(monkeypatch):
    waiting = threading.Event()
    waiting.set()
    stopping = threading.Event()
    stopping.set()
    turn = threading.Lock()
    monkeypatch.setattr(appmod, '_A_FIRST_OPEN_WAITS', waiting)
    monkeypatch.setattr(appmod, '_BACKGROUND_WARMS_STOPPING', stopping)
    monkeypatch.setattr(appmod, '_DATA_LIBRARY_TURN', turn)
    with appmod._the_data_library_turn() as acquired:
        assert acquired is False
        assert not turn.locked()
    assert waiting.is_set()


def test_abandoned_first_open_priority_expires_and_the_warm_can_finish(monkeypatch):
    waiting = threading.Event()
    waiting.set()
    stopping = threading.Event()
    turn = threading.Lock()
    monkeypatch.setattr(appmod, '_A_FIRST_OPEN_WAITS', waiting)
    monkeypatch.setattr(appmod, '_BACKGROUND_WARMS_STOPPING', stopping)
    monkeypatch.setattr(appmod, '_DATA_LIBRARY_TURN', turn)
    monkeypatch.setattr(appmod, '_DATA_LIBRARY_YIELD_S', 0)
    with appmod._the_data_library_turn() as acquired:
        assert acquired is True
        assert turn.locked()
        assert not waiting.is_set()
    assert not turn.locked()
