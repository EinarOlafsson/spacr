"""The failure paths of ``spacr.qt.timing``: the ones only a broken run takes.

Everything here is instrumentation, so every one of these paths exists to
keep a diagnostic from becoming the fault. Pinned in this file:

* the import-attribution frame walks -- the ``/spacr/`` frame they look for,
  the top of a stack they can run off, and an interpreter that refuses to
  hand out a frame at all;
* an interpreter with no ``ExtensionFileLoader`` to wrap;
* readiness probes whose widget, subscriber or event filter dies underneath
  them, which is the ordinary case when a screen is navigated away from
  while it is still being measured;
* the report and snapshot fields that only appear when a source of facts is
  missing -- no budget, no Qt, no ``resource``, no display, no preferences;
* ``begin`` refusing a spawn timestamp it cannot use, and refusing to run
  twice.

Nothing here leaves a wrapper on the loader or a probe in the register: the
``loader_restored`` and ``timing_on`` fixtures check that in teardown.
"""
from __future__ import annotations

import importlib.machinery
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("PySide6")

import shiboken6
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from spacr.qt import timing


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def timing_on(monkeypatch):
    """One empty, enabled instrumentation session with no process residue.

    Patching the module globals rather than reloading keeps every other
    reference to ``spacr.qt.timing`` in the session pointing at this module.
    """
    monkeypatch.setattr(timing, "ENABLED", True)
    monkeypatch.setattr(timing, "_START", time.perf_counter())
    monkeypatch.setattr(timing, "_EVENT_LOOP_STARTED_AT", None)
    monkeypatch.setattr(timing, "_SPANS", [])
    monkeypatch.setattr(timing, "_IMPORTS", [])
    monkeypatch.setattr(timing, "_STALLS", [])
    monkeypatch.setattr(timing, "_MARKS", [])
    monkeypatch.setattr(timing, "_READINESS", [])
    monkeypatch.setattr(timing, "_READY_CALLBACKS", [])
    monkeypatch.setattr(timing, "_ACTIVE_PROBES", [])
    yield
    for probe in list(timing._ACTIVE_PROBES):
        try:
            probe._retire()
        except (AttributeError, RuntimeError):
            pass


@pytest.fixture
def loader_restored():
    """Put ``exec_module`` back however the test ends.

    A leaked wrapper would keep recording into a list this file has thrown
    away, and would slow every import for the rest of the session.
    """
    source = importlib.machinery.SourceFileLoader.exec_module
    extension = importlib.machinery.ExtensionFileLoader.exec_module
    try:
        yield
    finally:
        importlib.machinery.SourceFileLoader.exec_module = source
        importlib.machinery.ExtensionFileLoader.exec_module = extension


def _function_from(filename: str, source: str, name: str):
    """Build a function whose stack frame claims to live in ``filename``.

    The attribution walks key off ``frame.f_code.co_filename``. Compiling the
    caller is the only way to put a chosen path on the stack without writing
    a module into the tree under test -- and this file's own name contains
    ``timing.py``, which the walks deliberately skip.
    """
    namespace: dict = {}
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102
    return namespace[name]


class _NoFrames:
    """``sys`` with no ``_getframe``, as on a runtime without one."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, item):
        return getattr(self._real, item)

    def _getframe(self, depth=0):
        raise ValueError("call stack is not deep enough")


class _WithoutSysconf:
    """``os`` on a platform with no ``sysconf`` page counts (Windows)."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, item):
        return getattr(self._real, item)

    def sysconf(self, name):
        raise OSError(f"unknown configuration name {name}")


class _PaintedButton(QPushButton):
    update_calls = 0

    def update(self, *args):  # noqa: D102 - Qt naming
        self.update_calls += 1
        super().update(*args)


def _page():
    root = QWidget()
    layout = QVBoxLayout(root)
    button = QPushButton("usable")
    layout.addWidget(button)
    return root, button


def _slow_module(tmp_path, name="a_deliberately_slow_probe"):
    """A real source file and loader whose execution clears the floor."""
    path = tmp_path / f"{name}.py"
    path.write_text("import time\ntime.sleep(0.02)\nVALUE = 1\n")
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    module = ModuleType(name)
    module.__file__ = str(path)
    return loader, module


# ---------------------------------------------------------------------------
# _ImportTimer.find_spec -- who asked for this module?
# ---------------------------------------------------------------------------

def test_the_finder_names_the_spacr_frame_that_asked_and_no_other(timing_on):
    """The attribution is the useful half: "3 s of torch, asked by qt/app.py".

    Both halves are driven here: a caller inside spaCR is named with its line
    number, and an identical caller outside it is not named at all -- the
    walk must not attribute an import to the first frame it happens to see.
    """
    body = "def ask(finder, name):\n    return finder.find_spec(name)\n"
    inside = _function_from("/nowhere/spacr/qt/asked_here.py", body, "ask")
    outside = _function_from("/nowhere/elsewhere/asked_here.py", body, "ask")

    named = timing._ImportTimer()
    inside(named, "a_module_no_process_has_imported")
    anonymous = timing._ImportTimer()
    outside(anonymous, "another_module_no_process_has_imported")

    assert named._pending[2] == "qt/asked_here.py:2"
    assert anonymous._pending[2] == ""
    assert anonymous._pending[0] == "another_module_no_process_has_imported"


def test_the_finder_walks_past_frames_but_stops_at_the_top_of_the_stack(
        timing_on):
    """A worker thread's stack ENDS, twelve frames or not.

    Left to run off it, the walk would dereference ``None``; an import timer
    that raises inside ``find_spec`` breaks importing itself. The same walk,
    given a spaCR frame two levels up, still finds it -- so the ``None``
    guard is a terminator and not a walk that never moved.
    """
    inner = _function_from(
        "/nowhere/elsewhere/inner_probe.py",
        "def inner(finder, name, box):\n"
        "    box.append(finder.find_spec(name))\n",
        "inner")
    outer = _function_from(
        "/nowhere/spacr/qt/outer_probe.py",
        "def outer(finder, name, box, inner):\n"
        "    inner(finder, name, box)\n",
        "outer")

    deep = timing._ImportTimer()
    box: list = []
    thread = threading.Thread(
        target=outer, args=(deep, "a_module_for_the_deep_walk", box, inner))
    thread.start()
    thread.join()

    shallow = timing._ImportTimer()
    thread = threading.Thread(
        target=inner, args=(shallow, "a_module_for_the_short_walk", box))
    thread.start()
    thread.join()

    assert deep._pending[2] == "qt/outer_probe.py:2"
    assert shallow._pending[2] == ""
    assert box == [None, None], "the finder must never resolve an import"


def test_a_runtime_that_will_not_give_a_frame_still_records_the_import(
        timing_on, monkeypatch):
    """``sys._getframe`` is a CPython detail; the timing must not depend on it.

    With it raising, the module name and start time are still captured and
    only the attribution is lost -- which is the whole point of the guard.
    """
    ask = _function_from(
        "/nowhere/spacr/qt/asked_here.py",
        "def ask(finder, name):\n    return finder.find_spec(name)\n", "ask")
    with_frames = timing._ImportTimer()
    ask(with_frames, "a_module_with_an_attributable_caller")

    without = timing._ImportTimer()
    monkeypatch.setattr(timing, "sys", _NoFrames(sys))
    ask(without, "a_module_with_no_attributable_caller")

    assert with_frames._pending[2] == "qt/asked_here.py:2"
    assert without._pending[0] == "a_module_with_no_attributable_caller"
    assert without._pending[2] == ""
    assert isinstance(without._pending[1], float)


def test_the_finders_note_hook_leaves_the_pending_attribution_alone(timing_on):
    """``note`` is the seam the loader wrapper took over; it must stay inert.

    A ``note`` that cleared or rewrote ``_pending`` would silently drop the
    attribution for whichever import was in flight.
    """
    finder = timing._ImportTimer()
    finder.find_spec("a_module_awaiting_its_attribution")
    pending = finder._pending

    assert finder.note() is None
    assert finder._pending == pending


# ---------------------------------------------------------------------------
# _install_import_timer
# ---------------------------------------------------------------------------

def test_an_interpreter_with_no_extension_loader_still_times_source_imports(
        loader_restored, monkeypatch):
    """``ExtensionFileLoader`` is not guaranteed to exist on every runtime.

    Losing C-extension timing is acceptable; losing source-import timing --
    or raising out of ``begin`` before the application starts -- is not.
    """
    source_before = importlib.machinery.SourceFileLoader.exec_module
    extension_before = importlib.machinery.ExtensionFileLoader.exec_module

    with monkeypatch.context() as absent:
        absent.setattr(timing, "_IMPORT_TIMER_INSTALLED", False)
        absent.delattr(importlib.machinery, "ExtensionFileLoader")
        timing._install_import_timer()
        assert (importlib.machinery.SourceFileLoader.exec_module
                is not source_before)

    assert (importlib.machinery.ExtensionFileLoader.exec_module
            is extension_before), "an absent class was wrapped anyway"

    # And with the class present the very same call does wrap it, so the
    # branch above is a real fork rather than a wrap that never happens.
    monkeypatch.setattr(timing, "_IMPORT_TIMER_INSTALLED", False)
    timing._install_import_timer()

    assert (importlib.machinery.ExtensionFileLoader.exec_module
            is not extension_before)


# ---------------------------------------------------------------------------
# the exec_module wrapper -- what triggered this import?
# ---------------------------------------------------------------------------

def test_a_slow_import_is_charged_to_the_spacr_frame_that_triggered_it(
        timing_on, loader_restored, monkeypatch, tmp_path):
    """The recorded ``by`` is what makes a slow import actionable."""
    monkeypatch.setattr(timing, "IMPORT_FLOOR_MS", 1.0)
    monkeypatch.setattr(timing, "_IMPORT_TIMER_INSTALLED", False)
    timing._install_import_timer()
    loader, module = _slow_module(tmp_path)
    # Two frames deep because the wrapper reads ``_getframe(2)``: in a real
    # import the frame between it and the requester is importlib's.
    run = _function_from(
        "/nowhere/spacr/qt/importer_probe.py",
        "def run(loader, module):\n"
        "    _call(loader, module)\n"
        "def _call(loader, module):\n"
        "    loader.exec_module(module)\n",
        "run")

    run(loader, module)

    assert [row["name"] for row in timing._IMPORTS] == [
        "a_deliberately_slow_probe"]
    assert timing._IMPORTS[0]["by"] == "qt/importer_probe.py:2"
    assert timing._IMPORTS[0]["took"] >= 0.019


def test_a_slow_import_off_the_top_of_a_worker_stack_is_still_recorded(
        timing_on, loader_restored, monkeypatch, tmp_path):
    """Preloader threads import; their stacks are four frames deep.

    The record must survive the walk reaching the end of that stack, keeping
    the module and its cost even when nothing can be blamed for it.
    """
    monkeypatch.setattr(timing, "IMPORT_FLOOR_MS", 1.0)
    monkeypatch.setattr(timing, "_IMPORT_TIMER_INSTALLED", False)
    timing._install_import_timer()
    loader, module = _slow_module(tmp_path)
    attributable = _function_from(
        "/nowhere/spacr/qt/thread_probe.py",
        "def run(loader, module):\n"
        "    _call(loader, module)\n"
        "def _call(loader, module):\n"
        "    loader.exec_module(module)\n",
        "run")
    anonymous = _function_from(
        "/nowhere/elsewhere/thread_probe.py",
        "def run(loader, module):\n    loader.exec_module(module)\n", "run")

    for target in (attributable, anonymous):
        thread = threading.Thread(target=target, args=(loader, module),
                                  name="preloader-probe")
        thread.start()
        thread.join()

    assert len(timing._IMPORTS) == 2
    assert timing._IMPORTS[0]["by"] == "qt/thread_probe.py:2"
    assert timing._IMPORTS[1]["by"] == ""
    assert timing._IMPORTS[1]["thread"] == "preloader-probe"


def test_a_runtime_that_will_not_give_a_frame_still_records_the_cost(
        timing_on, loader_restored, monkeypatch, tmp_path):
    """Attribution is best-effort; the duration is not."""
    monkeypatch.setattr(timing, "IMPORT_FLOOR_MS", 1.0)
    monkeypatch.setattr(timing, "_IMPORT_TIMER_INSTALLED", False)
    timing._install_import_timer()
    loader, module = _slow_module(tmp_path)
    run = _function_from(
        "/nowhere/spacr/qt/importer_probe.py",
        "def run(loader, module):\n"
        "    _call(loader, module)\n"
        "def _call(loader, module):\n"
        "    loader.exec_module(module)\n",
        "run")

    run(loader, module)
    monkeypatch.setattr(timing, "sys", _NoFrames(sys))
    run(loader, module)

    assert timing._IMPORTS[0]["by"] == "qt/importer_probe.py:2"
    assert timing._IMPORTS[1]["by"] == ""
    assert timing._IMPORTS[1]["took"] >= 0.019


# ---------------------------------------------------------------------------
# probes that die under the event loop
# ---------------------------------------------------------------------------

def test_a_probe_that_dies_when_the_loop_starts_is_dropped_not_raised_through(
        timing_on):
    """``event_loop_started`` runs once, on the path to the first frame.

    A deleted widget behind one probe must not stop the loop from telling the
    others -- and a probe that has already dropped its own registration must
    not make the drop fail either.
    """
    class _Dead:
        def event_loop_started(self):
            raise RuntimeError("Internal C++ object already deleted.")

    class _DeadAndAlreadyGone:
        def event_loop_started(self):
            timing._ACTIVE_PROBES.remove(self)
            raise RuntimeError("Internal C++ object already deleted.")

    class _Healthy:
        def __init__(self):
            self.told = 0

        def event_loop_started(self):
            self.told += 1

    dead, vanished, healthy = _Dead(), _DeadAndAlreadyGone(), _Healthy()
    timing._ACTIVE_PROBES.extend([dead, vanished, healthy])

    timing.event_loop_started()

    assert healthy.told == 1
    assert timing._ACTIVE_PROBES == [healthy]
    assert timing._EVENT_LOOP_STARTED_AT is not None


def test_cancelling_by_detail_leaves_the_same_screen_under_another_detail(
        timing_on):
    """Two modules share a report name and differ only by their detail.

    Matching on detail is what lets one of them give up without retiring the
    readiness measurement the other is still waiting on.
    """
    class _Probe:
        def __init__(self, detail):
            self.report_name = "interactive module"
            self.report_detail = detail
            self.retired = False

        def _retire(self):
            self.retired = True

    wanted = _Probe("classify")
    other = _Probe("measure")
    timing._ACTIVE_PROBES.extend([wanted, other])

    retired = timing.cancel_interactive(detail="classify")

    assert retired == 1
    assert wanted.retired is True
    assert other.retired is False


def test_cancelling_a_probe_that_died_and_deregistered_itself_still_counts(
        timing_on):
    """Cancel races the probe's own retirement; both may reach the register.

    A probe destroyed with its widget raises from ``_retire``. If it had
    already dropped its registration on the way down, the tidy-up that
    follows must not raise in turn -- and a probe still registered must
    genuinely be dropped, or the next screen would inherit it.
    """
    class _DeadButRegistered:
        report_name = "gone"
        report_detail = ""

        def _retire(self):
            raise RuntimeError("Internal C++ object already deleted.")

    class _DeadAndDeregistered:
        report_name = "gone"
        report_detail = ""

        def _retire(self):
            timing._ACTIVE_PROBES.remove(self)
            raise RuntimeError("Internal C++ object already deleted.")

    class _Live:
        report_name = "still here"
        report_detail = ""

        def __init__(self):
            self.retired = False

        def _retire(self):
            self.retired = True

    registered = _DeadButRegistered()
    deregistered = _DeadAndDeregistered()
    live = _Live()
    timing._ACTIVE_PROBES.extend([registered, deregistered, live])

    assert timing.cancel_interactive(name="gone") == 2
    assert timing._ACTIVE_PROBES == [live]
    assert live.retired is False


# ---------------------------------------------------------------------------
# watch_interactive
# ---------------------------------------------------------------------------

def test_a_screen_that_is_itself_a_control_counts_itself(qtbot, timing_on):
    """A module page can BE a control -- a single combo box, a single button.

    Skipping the root would leave such a page with no observable control and
    no readiness record at all.
    """
    button = QPushButton("the whole screen")
    qtbot.addWidget(button)
    plain_root, child = _page()
    qtbot.addWidget(plain_root)

    itself = timing.watch_interactive(button, "interactive module", "button")
    container = timing.watch_interactive(
        plain_root, "interactive module", "container")

    assert itself.controls == (button,)
    assert container.controls == (child,)
    assert plain_root not in container.controls


def test_only_visible_controls_are_repainted_when_the_loop_starts(
        qtbot, timing_on):
    """Forcing a repaint on a hidden control buys nothing and costs a frame.

    Qt discards it, so the only effect of dropping the visibility test would
    be work done on every hidden widget of every screen at start-up.
    """
    root = QWidget()
    layout = QVBoxLayout(root)
    shown = _PaintedButton("shown")
    hidden = _PaintedButton("hidden")
    layout.addWidget(shown)
    layout.addWidget(hidden)
    qtbot.addWidget(root)
    timing.watch_interactive(root, "interactive module", "visibility")
    root.show()
    hidden.hide()
    qtbot.wait(20)
    shown_before = shown.update_calls
    hidden_before = hidden.update_calls

    timing.event_loop_started()

    assert shown.isVisible() and not hidden.isVisible()
    assert shown.update_calls == shown_before + 1
    assert hidden.update_calls == hidden_before


def test_a_control_deleted_before_the_loop_starts_retires_the_probe(
        qtbot, timing_on):
    """Navigating away during start-up deletes the page being measured.

    The probe must retire itself rather than raise out of the one callback
    every other probe is waiting behind.
    """
    root, button = _page()
    qtbot.addWidget(root)
    probe = timing.watch_interactive(root, "interactive module", "deleted")
    root.show()
    qtbot.wait(20)
    assert probe in timing._ACTIVE_PROBES

    shiboken6.delete(button)
    timing.event_loop_started()

    assert probe.done is True
    assert probe not in timing._ACTIVE_PROBES
    assert timing._READINESS == []


def test_a_deleted_control_is_not_counted_among_the_usable_ones(
        qtbot, timing_on, monkeypatch):
    """The count in the report is "controls the user could operate".

    A control destroyed between its paint and the settle is not one of them,
    and asking a deleted object for its size raises rather than answering.
    """
    root = QWidget()
    layout = QVBoxLayout(root)
    survivor = QPushButton("survivor")
    doomed = QPushButton("doomed")
    layout.addWidget(survivor)
    layout.addWidget(doomed)
    qtbot.addWidget(root)
    probe = timing.watch_interactive(root, "interactive module", "usable")
    root.show()
    qtbot.wait(30)
    assert probe.painted_controls == {id(survivor), id(doomed)}

    shiboken6.delete(doomed)
    # The settle is normally queued by a paint; calling it directly is the
    # only way to place the deletion inside the window between the two.
    monkeypatch.setattr(timing, "_EVENT_LOOP_STARTED_AT", time.perf_counter())
    probe._settle()

    entry = timing._READINESS[-1]
    assert entry["usable_controls"] == 1
    assert entry["painted_usable_controls"] == 1
    assert entry["controls"] == [survivor.objectName() or "QPushButton"]


def test_a_screen_destroyed_before_it_settles_retires_rather_than_reports(
        qtbot, timing_on, monkeypatch):
    """A readiness record for a window that no longer exists is a lie.

    The healthy probe in the same page proves the settle would otherwise
    have produced one.
    """
    root, _button = _page()
    qtbot.addWidget(root)
    healthy = timing.watch_interactive(root, "still here", "probe")
    doomed = timing.watch_interactive(root, "gone", "probe")
    root.show()
    qtbot.wait(30)
    monkeypatch.setattr(timing, "_EVENT_LOOP_STARTED_AT", time.perf_counter())

    healthy._settle()
    ghost = QWidget()
    shiboken6.delete(ghost)
    # Substituting a deleted widget for the root is the only deterministic
    # way to reach the window between the queued settle and its callback.
    doomed.root = ghost
    doomed._settle()

    assert [entry["name"] for entry in timing._READINESS] == ["still here"]
    assert doomed.done is True
    assert doomed not in timing._ACTIVE_PROBES


def test_a_readiness_subscriber_that_raises_does_not_stop_the_next_one(
        qtbot, timing_on, monkeypatch):
    """Instrumentation may never make navigation fail.

    The raising subscriber is registered first, so the second one only hears
    about the screen if the failure was contained.
    """
    root, _button = _page()
    qtbot.addWidget(root)
    seen: list = []

    def _explodes(_entry):
        raise ValueError("a subscriber that cannot cope")

    timing.subscribe_readiness(_explodes)
    timing.subscribe_readiness(seen.append)
    probe = timing.watch_interactive(root, "interactive module", "subscribers")
    root.show()
    qtbot.wait(30)
    monkeypatch.setattr(timing, "_EVENT_LOOP_STARTED_AT", time.perf_counter())

    probe._settle()

    assert [entry["name"] for entry in seen] == ["interactive module"]
    assert len(timing._READINESS) == 1


def test_retiring_a_probe_twice_does_no_work_the_second_time(qtbot, timing_on):
    """Cancel and settle can both retire the same probe.

    Without the latch the second pass would re-run the teardown -- and would
    remove whatever now stands where the probe used to be in the register.
    """
    root, _button = _page()
    qtbot.addWidget(root)
    probe = timing.watch_interactive(root, "interactive module", "twice")

    probe._retire()
    assert probe.done is True
    assert probe not in timing._ACTIVE_PROBES

    # A stale re-registration stands in for "something else is in the
    # register now": the latch must return before touching it.
    timing._ACTIVE_PROBES.append(probe)
    probe._retire()

    assert timing._ACTIVE_PROBES == [probe]


def test_a_probe_whose_registration_is_already_gone_still_retires(
        qtbot, timing_on):
    """A new timing session empties the register under a live probe."""
    root, _button = _page()
    qtbot.addWidget(root)
    probe = timing.watch_interactive(root, "interactive module", "orphan")
    assert probe in timing._ACTIVE_PROBES

    timing._ACTIVE_PROBES.remove(probe)
    probe._retire()

    assert probe.done is True
    assert timing._ACTIVE_PROBES == []


def test_a_root_that_refuses_an_event_filter_still_watches_its_controls(
        qtbot, timing_on):
    """Half an observer beats none: the contract is a painted CONTROL.

    A root already being torn down raises from ``installEventFilter``; the
    child controls are still installable, and readiness still arrives -- with
    ``root_painted`` false, because that root's paints were never seen.
    """
    class _RefusesEventFilter(QWidget):
        def installEventFilter(self, watcher):  # noqa: D102 - Qt naming
            raise RuntimeError("Internal C++ object already deleted.")

    root = _RefusesEventFilter()
    layout = QVBoxLayout(root)
    button = QPushButton("usable")
    layout.addWidget(button)
    qtbot.addWidget(root)

    probe = timing.watch_interactive(root, "interactive module", "no filter")
    root.show()
    qtbot.wait(30)
    timing.event_loop_started()
    qtbot.waitUntil(lambda: len(timing._READINESS) == 1, timeout=2000)

    entry = timing._READINESS[0]
    assert probe in timing._ACTIVE_PROBES or probe.done is True
    assert entry["painted_usable_controls"] == 1
    assert entry["root_painted"] is False


# ---------------------------------------------------------------------------
# report and snapshot with a source of facts missing
# ---------------------------------------------------------------------------

def test_a_readiness_entry_with_no_budget_gets_no_verdict(timing_on):
    """Only measurements taken against a budget may be graded.

    A diagnostic ``watch_interactive`` call passes no budget; printing "OK"
    for it would invent a release contract that was never asked for.
    """
    timing._READINESS.extend([
        {"at": 1.0, "duration_s": 0.5, "name": "diagnostic page",
         "detail": "no budget", "budget_s": None, "within_budget": None,
         "painted_usable_controls": 3},
        {"at": 2.0, "duration_s": 1.0, "name": "interactive module",
         "detail": "probe", "budget_s": 10.0, "within_budget": True,
         "painted_usable_controls": 2},
    ])

    lines = timing.report().splitlines()

    ungraded = [line for line in lines if "diagnostic page" in line][0]
    graded = [line for line in lines if "interactive module" in line][0]
    assert ungraded.endswith("3 painted control(s)")
    assert graded.endswith("OK")


def test_peak_memory_prefers_the_windows_peak_working_set(monkeypatch):
    """``peak_wset`` is a peak; ``rss`` is only the current value.

    Reporting the current RSS as a peak on a platform that has the real one
    would understate every measurement taken after a spike.
    """
    monkeypatch.setitem(sys.modules, "resource", None)
    fake = ModuleType("psutil")
    info = SimpleNamespace(peak_wset=512 * 1024 * 1024, rss=64 * 1024 * 1024)
    fake.Process = lambda: SimpleNamespace(memory_info=lambda: info)
    monkeypatch.setitem(sys.modules, "psutil", fake)

    assert timing._peak_rss_mb() == 512.0

    del info.peak_wset
    assert timing._peak_rss_mb() == 64.0


def test_a_torch_allocator_that_raises_reports_nothing_rather_than_zero(
        monkeypatch):
    """Zero allocated bytes and "could not ask" are different facts.

    A driver-level failure reported as 0 MB would read as a clean run in the
    artifact.
    """
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_initialized=lambda: True,
        memory_allocated=lambda: 4 * 1024 * 1024,
        max_memory_allocated=lambda: 8 * 1024 * 1024,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert timing._gpu_memory_mb() == {
        "allocated_mb": 4.0, "peak_allocated_mb": 8.0}

    def _no_driver():
        raise RuntimeError("CUDA driver could not be reached")

    torch.cuda = SimpleNamespace(is_initialized=_no_driver)

    assert timing._gpu_memory_mb() == {
        "allocated_mb": None, "peak_allocated_mb": None}


def test_total_memory_falls_back_to_psutil_and_then_gives_up(monkeypatch):
    """``os.sysconf`` has no ``SC_PHYS_PAGES`` on Windows.

    psutil is the fallback where it is installed; where it is not, the field
    is honestly absent rather than guessed.
    """
    monkeypatch.setattr(timing, "os", _WithoutSysconf(timing.os))
    fake = ModuleType("psutil")
    fake.virtual_memory = lambda: SimpleNamespace(total=8 * 1024 ** 3)
    monkeypatch.setitem(sys.modules, "psutil", fake)

    assert timing._hardware_profile()["total_memory_mb"] == 8192.0

    monkeypatch.setitem(sys.modules, "psutil", None)

    assert timing._hardware_profile()["total_memory_mb"] is None


def test_a_display_that_cannot_be_queried_reports_no_displays(monkeypatch):
    """A screen unplugged mid-enumeration must not lose the whole snapshot.

    Half a display list is worse than none -- it would be read as the machine
    having had one monitor.
    """
    screen = SimpleNamespace(
        name=lambda: "probe-0",
        geometry=lambda: SimpleNamespace(width=lambda: 1920,
                                         height=lambda: 1080),
        devicePixelRatio=lambda: 2.0,
        refreshRate=lambda: 60.0,
    )
    app = SimpleNamespace(screens=lambda: [screen],
                          platformName=lambda: "probe")
    widgets = ModuleType("PySide6.QtWidgets")
    widgets.QApplication = SimpleNamespace(instance=lambda: app)
    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", widgets)

    profile = timing._hardware_profile()
    assert profile["qt_platform"] == "probe"
    assert profile["displays"] == [{
        "name": "probe-0", "logical_width": 1920, "logical_height": 1080,
        "device_pixel_ratio": 2.0, "refresh_hz": 60.0,
    }]

    def _unplugged():
        raise RuntimeError("the screen list changed under enumeration")

    app.screens = _unplugged

    assert timing._hardware_profile()["displays"] == []


def test_a_preferences_module_that_raises_leaves_the_level_unknown(
        monkeypatch):
    """The performance level is read from whatever is already imported.

    A half-initialised preferences module must not take the snapshot with it.
    """
    preferences = ModuleType("spacr.qt.preferences")
    preferences.get_performance_level = lambda: "high"
    monkeypatch.setitem(sys.modules, "spacr.qt.preferences", preferences)
    assert timing._hardware_profile()["performance_level"] == "high"

    def _not_ready():
        raise RuntimeError("preferences are not loaded yet")

    preferences.get_performance_level = _not_ready

    assert timing._hardware_profile()["performance_level"] is None


def test_a_snapshot_taken_without_qt_loaded_carries_no_qt_version(monkeypatch):
    """``snapshot`` never imports Qt; it reports what the process already has.

    A benchmark that failed before Qt was imported must still produce a
    readable artifact, and one whose Qt cannot answer must not claim a
    version it did not get.
    """
    assert isinstance(timing.snapshot()["environment"]["qt"], str)

    monkeypatch.delitem(sys.modules, "PySide6.QtCore")
    assert timing.snapshot()["environment"]["qt"] is None

    core = ModuleType("PySide6.QtCore")

    def _refuses():
        raise RuntimeError("Qt was unloaded")

    core.qVersion = _refuses
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", core)

    assert timing.snapshot()["environment"]["qt"] is None


# ---------------------------------------------------------------------------
# begin
# ---------------------------------------------------------------------------

def test_begin_dates_the_clock_from_this_module_and_arms_the_import_timer(
        timing_on, loader_restored, monkeypatch):
    """An ordinary ``SPACR_TIMING=1 spacr`` has no parent timestamp to use."""
    monkeypatch.delenv("SPACR_TIMING_PROCESS_START", raising=False)
    monkeypatch.setattr(timing, "IMPORT_TIMING_ENABLED", True)
    monkeypatch.setattr(timing, "_IMPORT_TIMER_INSTALLED", False)
    monkeypatch.setattr(timing.begin, "_done", False, raising=False)
    origin = timing._START
    before = importlib.machinery.SourceFileLoader.exec_module

    timing.begin()

    assert timing._START == origin
    assert timing._MARKS[-1]["name"] == "timing started"
    assert timing._MARKS[-1]["detail"] == "timing module import"
    assert importlib.machinery.SourceFileLoader.exec_module is not before


def test_begin_refuses_to_move_the_clock_a_second_time(timing_on, monkeypatch):
    """Two ``begin`` calls in one process -- a test session, an embedded run.

    The second is given a spawn timestamp that WOULD move the origin, so the
    latch is what keeps every measurement so far comparable.
    """
    monkeypatch.setattr(timing, "IMPORT_TIMING_ENABLED", False)
    monkeypatch.setattr(timing.begin, "_done", False, raising=False)
    monkeypatch.delenv("SPACR_TIMING_PROCESS_START", raising=False)
    timing.begin()
    origin = timing._START

    monkeypatch.setenv("SPACR_TIMING_PROCESS_START", str(time.time() - 2.0))
    timing.begin()

    assert timing._START == origin
    assert [mark["name"] for mark in timing._MARKS] == ["timing started"]


def test_a_spawn_timestamp_that_cannot_be_parsed_is_ignored(
        timing_on, monkeypatch):
    """The variable comes from a shell; it can be anything at all."""
    monkeypatch.setattr(timing, "IMPORT_TIMING_ENABLED", False)
    monkeypatch.setattr(timing.begin, "_done", False, raising=False)
    monkeypatch.setenv("SPACR_TIMING_PROCESS_START", "not-a-timestamp")
    origin = timing._START

    timing.begin()

    assert timing._START == origin
    assert timing._MARKS[-1]["detail"] == "timing module import"


def test_a_spawn_timestamp_from_another_era_is_refused(timing_on, monkeypatch):
    """A stale or future value would date the run hours before it started.

    The same code path accepts a plausible one, so the window is a filter and
    not a switch that is always off.
    """
    monkeypatch.setattr(timing, "IMPORT_TIMING_ENABLED", False)
    monkeypatch.setattr(timing.begin, "_done", False, raising=False)
    monkeypatch.setenv("SPACR_TIMING_PROCESS_START", "1.0")
    origin = timing._START

    timing.begin()
    assert timing._START == origin
    assert timing._MARKS[-1]["detail"] == "timing module import"

    monkeypatch.setattr(timing.begin, "_done", False, raising=False)
    monkeypatch.setenv("SPACR_TIMING_PROCESS_START", str(time.time() - 2.0))

    timing.begin()

    assert timing._START < origin
    assert timing._MARKS[-1]["detail"] == "benchmark process spawn"
