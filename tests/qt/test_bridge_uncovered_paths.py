"""The bridge's edges: stream buffering, show routing, parked threads, run exits.

Everything here is a path the run loop takes when something around it is
already wrong — a library that prints a kilobyte without a newline, a
figure whose manager has gone, a thread that will not stop, a pipeline that
calls ``sys.exit(1)``. None of them may cost the caller its ``finished``
signal, and none may leave a process-wide hook installed twice.
"""
from __future__ import annotations

import gc
import logging
import sys
import threading

import pytest

pytest.importorskip("PySide6")

import shiboken6  # noqa: E402
from PySide6.QtCore import QThread  # noqa: E402

from spacr.cancellation import PipelineCancelled  # noqa: E402
from spacr.qt import bridge as B  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _isolated_worker_journal(monkeypatch, tmp_path):
    """GUI workers must not write test runs into the user's real journal."""
    from spacr import run_journal

    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: root)
    return root


# ---------------------------------------------------------------------------
# _StreamRedirector
# ---------------------------------------------------------------------------

def test_a_kilobyte_without_a_newline_is_shown_rather_than_held():
    """Cellpose prints its model download with no newline for a long time.

    A pure emit-on-newline redirector holds that hostage and the app looks
    hung, so the buffer is emitted once it passes the chunk cap.
    """
    received = []
    redirect = B._StreamRedirector(received.append)

    redirect.write("x" * (B._StreamRedirector._MAX_BUF_CHARS - 1))
    assert received == [], "emitted before the cap was reached"

    redirect.write("yz")

    assert len(received) == 1
    assert len(received[0]) == B._StreamRedirector._MAX_BUF_CHARS + 1
    assert received[0].endswith("yz")
    # The buffer was handed over, not copied: the next flush has nothing.
    redirect.flush()
    assert len(received) == 1


def test_the_idle_pump_surfaces_a_short_line_that_never_gets_its_newline():
    received = []
    redirect = B._StreamRedirector(received.append)
    redirect.write("Downloading cpsam: 12%")

    redirect.idle_flush()

    assert received == ["Downloading cpsam: 12%"]
    redirect.idle_flush()
    assert received == ["Downloading cpsam: 12%"], "an empty buffer emitted"


def test_a_console_that_raises_never_reaches_the_writing_thread():
    """The writer is a pipeline; a slot that throws must not break its print."""
    def _explode(_chunk):
        raise RuntimeError("the console widget is gone")

    redirect = B._StreamRedirector(_explode)

    assert redirect.write("a line\n") == len("a line\n")


# ---------------------------------------------------------------------------
# _ThreadStreamRouter
# ---------------------------------------------------------------------------

def test_unregistering_a_stream_twice_leaves_the_router_consistent():
    """Teardown can run twice; the second must not raise or evict a peer."""
    original = B._StreamRedirector(lambda _s: None)
    router = B._ThreadStreamRouter(original)
    first = B._StreamRedirector(lambda _s: None)
    second = B._StreamRedirector(lambda _s: None)
    router.register(first)
    router.register(second)

    router.unregister(second)
    router.unregister(second)

    assert router.has_targets() is True, "the surviving stream was evicted"
    assert router._target() is first

    router.unregister(first)
    assert router.has_targets() is False
    assert router._target() is original


# ---------------------------------------------------------------------------
# plt.show routing
# ---------------------------------------------------------------------------

class _FakePlt:
    """Stands in for ``matplotlib.pyplot`` for the routing bookkeeping only."""

    def __init__(self):
        self.calls = []
        self.show = self._original_show

    def _original_show(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "original"


@pytest.fixture
def show_routing():
    """Save and restore the process-wide ``plt.show`` routing state."""
    saved = (dict(B._MPL_SHOW_TARGETS), B._MPL_ORIGINAL_SHOW, B._MPL_MODULE)
    B._MPL_SHOW_TARGETS.clear()
    B._MPL_ORIGINAL_SHOW = None
    B._MPL_MODULE = None
    try:
        yield
    finally:
        B._MPL_SHOW_TARGETS.clear()
        B._MPL_SHOW_TARGETS.update(saved[0])
        B._MPL_ORIGINAL_SHOW = saved[1]
        B._MPL_MODULE = saved[2]


def test_a_worker_thread_with_no_capture_drops_show_instead_of_opening_a_loop(
        show_routing, caplog):
    """``plt.show`` off the GUI thread would start a Qt event loop there."""
    plt = _FakePlt()
    captured = []
    B._register_matplotlib_show(plt, captured.append)

    answers = {}

    def _in_a_worker():
        B._unregister_matplotlib_show(captured.append)
        with caplog.at_level(logging.DEBUG, logger=B.LOG.name):
            answers["result"] = B._matplotlib_show_router()

    worker = threading.Thread(target=_in_a_worker, name="probe-worker")
    worker.start()
    worker.join(5)

    assert answers["result"] is None
    assert plt.calls == [], "the original show ran off the GUI thread"
    assert "dropped rather than" in caplog.text


def test_the_main_thread_with_no_capture_still_shows_the_figure(show_routing):
    """Routing is per thread: a worker's capture must not swallow a GUI show."""
    plt = _FakePlt()

    def _register_from_a_worker():
        B._register_matplotlib_show(plt, lambda *a, **k: "captured")

    worker = threading.Thread(target=_register_from_a_worker, name="probe-owner")
    worker.start()
    worker.join(5)
    assert plt.show is B._matplotlib_show_router

    assert B._matplotlib_show_router(1, key=2) == "original"
    assert plt.calls == [((1,), {"key": 2})]


def test_a_second_capture_on_the_same_thread_does_not_relearn_the_original(
        show_routing):
    """The saved original must survive nested registrations."""
    plt = _FakePlt()
    original = plt.show
    B._register_matplotlib_show(plt, lambda *a, **k: "outer")

    B._register_matplotlib_show(plt, lambda *a, **k: "inner")

    assert B._MPL_ORIGINAL_SHOW is original, "the router was saved as the original"
    assert plt.show is B._matplotlib_show_router
    assert B._matplotlib_show_router() == "inner", "the newest capture is not on top"


def test_releasing_the_inner_capture_hands_show_back_to_the_outer_one(
        show_routing):
    plt = _FakePlt()
    inner = lambda *a, **k: "inner"          # noqa: E731
    B._register_matplotlib_show(plt, lambda *a, **k: "outer")
    B._register_matplotlib_show(plt, inner)

    B._unregister_matplotlib_show(inner)

    assert B._matplotlib_show_router() == "outer"
    assert plt.show is B._matplotlib_show_router, "show was restored too early"
    assert B._MPL_ORIGINAL_SHOW is not None


def test_releasing_a_capture_that_was_never_registered_changes_nothing(
        show_routing):
    plt = _FakePlt()
    outer = lambda *a, **k: "outer"          # noqa: E731
    B._register_matplotlib_show(plt, outer)

    B._unregister_matplotlib_show(lambda *a, **k: "never registered")

    assert B._matplotlib_show_router() == "outer"
    assert plt.show is B._matplotlib_show_router


def test_a_show_that_somebody_else_replaced_is_left_alone_on_release(
        show_routing):
    """Restoring blindly would clobber whatever took the slot after us."""
    plt = _FakePlt()
    capture = lambda *a, **k: "captured"     # noqa: E731
    B._register_matplotlib_show(plt, capture)
    theirs = lambda *a, **k: "theirs"        # noqa: E731
    plt.show = theirs

    B._unregister_matplotlib_show(capture)

    assert plt.show is theirs
    assert B._MPL_ORIGINAL_SHOW is None, "the routing state was not released"
    assert B._MPL_MODULE is None


# ---------------------------------------------------------------------------
# PauseGate
# ---------------------------------------------------------------------------

def test_pausing_an_already_paused_gate_does_not_restart_its_clock():
    gate = B.PauseGate()
    gate.pause()
    first = gate._paused_since

    gate.pause()

    assert gate._paused_since == first
    assert gate.is_paused() is True
    assert gate.paused_for() >= 0.0


def test_waiting_on_a_running_gate_returns_at_once():
    gate = B.PauseGate()

    assert gate.wait_if_paused(timeout=5.0) is True
    assert gate.paused_for() == 0.0


def test_a_paused_gate_releases_the_waiter_when_it_resumes(qtbot):
    gate = B.PauseGate()
    gate.pause()
    released = []

    def _wait():
        released.append(gate.wait_if_paused(timeout=5.0))

    waiter = threading.Thread(target=_wait, name="paused-probe")
    waiter.start()
    gate.resume()
    waiter.join(5)

    assert released == [True]
    assert gate.is_paused() is False


# ---------------------------------------------------------------------------
# RunHandle
# ---------------------------------------------------------------------------

@pytest.fixture
def handle():
    """A handle over a real worker and an unstarted thread, never registered."""
    worker = B.PipelineWorker(lambda settings: None, {})
    made = B.RunHandle("measure", worker, QThread())
    yield made
    made.worker = None
    made.thread = None


def test_a_handle_reports_the_workers_own_pause_support_and_gate(handle):
    assert handle.supports_pause is False, "no shipped pipeline opts in"
    assert handle.gate is handle.worker.gate
    assert handle.elapsed() >= 0.0


def test_a_pausable_entry_point_is_reported_as_pausable():
    def _fn(settings):
        return None

    B.pausable(_fn)
    worker = B.PipelineWorker(_fn, {})

    assert worker.supports_pause is True
    assert B.RunHandle("measure", worker, QThread()).supports_pause is True


def test_cancelling_a_retired_handle_is_a_no_op(handle, monkeypatch):
    monkeypatch.setattr(B, "_REGISTRY", B.RunRegistry())
    handle.retire()
    assert handle.worker is None and handle.thread is None

    handle.request_cancel("shutdown")  # must not raise

    assert handle.is_running() is False


def test_a_handle_with_no_progress_line_reports_no_fraction(handle):
    assert handle.fraction() is None


def test_a_progress_line_becomes_a_fraction(handle):
    handle._on_line("Progress: 41/96 fields\n")

    assert handle.progress == (41, 96)
    assert handle.fraction() == pytest.approx(41 / 96)
    assert handle.last_line == "Progress: 41/96 fields"


def test_a_progress_line_with_no_total_reports_no_fraction(handle):
    handle._on_line("Progress: 3/0\n")

    assert handle.progress == (3, 0)
    assert handle.fraction() is None


def test_a_fraction_is_clamped_to_the_unit_interval(handle):
    handle._on_line("Progress: 120/96\n")

    assert handle.fraction() == 1.0


# ---------------------------------------------------------------------------
# RunRegistry
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_registry(monkeypatch):
    """A registry of this test's own, so nothing else's jobs are cancelled."""
    made = B.RunRegistry()
    monkeypatch.setattr(B, "_REGISTRY", made)
    return made


def test_cancel_all_walks_past_a_job_whose_thread_already_stopped(
        isolated_registry, handle):
    """The thread was never started, so there is nothing to wait for."""
    isolated_registry.register(handle)

    still_running = isolated_registry.cancel_all(timeout_ms=200)

    assert still_running == []
    assert handle.worker.cancel_token.cancelled is True


def test_cancel_all_ignores_a_handle_whose_thread_reference_has_gone(
        isolated_registry, handle):
    isolated_registry.register(handle)
    handle.thread = None

    assert isolated_registry.cancel_all(timeout_ms=200) == []


def test_clearing_the_registry_empties_it_and_announces_the_change(
        isolated_registry, handle, qtbot):
    isolated_registry.register(handle)
    assert isolated_registry.is_busy() is True

    with qtbot.waitSignal(isolated_registry.changed, timeout=2000):
        isolated_registry.clear()

    assert isolated_registry.active() == []
    assert isolated_registry.is_busy() is False


def test_clearing_an_empty_registry_announces_nothing(isolated_registry):
    seen = []
    isolated_registry.changed.connect(lambda: seen.append(1))

    isolated_registry.clear()

    assert seen == []


# ---------------------------------------------------------------------------
# Parked threads
# ---------------------------------------------------------------------------

@pytest.fixture
def parked():
    """Save and restore the process-wide parked-thread list."""
    with B._PARKED_LOCK:
        saved = list(B._PARKED_THREADS)
        B._PARKED_THREADS.clear()
    try:
        yield B._PARKED_THREADS
    finally:
        with B._PARKED_LOCK:
            B._PARKED_THREADS[:] = saved


def test_a_parked_thread_that_has_finished_is_released(parked):
    thread = QThread()
    parked.append((thread, object()))

    assert B.wait_for_parked_threads(timeout_ms=500) == 0
    assert B.parked_thread_count() == 0


def test_a_parked_thread_whose_wrapper_has_gone_is_released(parked):
    thread = QThread()
    parked.append((thread, object()))
    shiboken6.delete(thread)

    assert B.wait_for_parked_threads(timeout_ms=500) == 0
    assert B.parked_thread_count() == 0


def test_the_exit_wait_is_one_shared_budget_not_one_per_thread(parked):
    """With the budget already spent, the remaining threads are not waited on."""
    waited = []

    class _Slow:
        def wait(self, ms):
            waited.append(ms)
            return True

        def isRunning(self):
            return False

    parked.extend([(_Slow(), object()), (_Slow(), object())])

    assert B.wait_for_parked_threads(timeout_ms=0) == 0
    assert waited == [], "a thread was waited on after the budget expired"


def test_the_exit_hook_says_how_many_threads_outlived_the_process(
        monkeypatch, caplog):
    """Qt treats destroying a running QThread as fatal, so this is logged loudly."""
    monkeypatch.setattr(B, "wait_for_parked_threads", lambda: 3)

    with caplog.at_level(logging.ERROR, logger=B.LOG.name):
        B._drain_parked_threads_at_exit()

    assert "3 worker thread(s) are still running" in caplog.text


def test_the_exit_hook_stays_quiet_when_everything_stopped(monkeypatch, caplog):
    monkeypatch.setattr(B, "wait_for_parked_threads", lambda: 0)

    with caplog.at_level(logging.ERROR, logger=B.LOG.name):
        B._drain_parked_threads_at_exit()

    assert "still running as the process exits" not in caplog.text


def test_the_exit_hook_is_registered_once_however_many_threads_are_parked(
        monkeypatch):
    registered = []
    monkeypatch.setattr(B.atexit, "register", registered.append)
    was_installed = B._PARKED_EXIT_HOOK_INSTALLED
    B._PARKED_EXIT_HOOK_INSTALLED = False
    try:
        B._install_parked_exit_hook()
        B._install_parked_exit_hook()
    finally:
        B._PARKED_EXIT_HOOK_INSTALLED = was_installed

    assert registered == [B._drain_parked_threads_at_exit]


# ---------------------------------------------------------------------------
# drain_thread
# ---------------------------------------------------------------------------

def test_draining_a_thread_that_never_started_succeeds_without_parking(parked):
    thread = QThread()

    assert B.drain_thread(thread) is True
    assert B.parked_thread_count() == 0


def test_draining_nothing_succeeds(parked):
    assert B.drain_thread(None) is True


def test_draining_a_thread_whose_wrapper_has_gone_succeeds(parked):
    thread = QThread()
    shiboken6.delete(thread)

    assert B.drain_thread(thread) is True
    assert B.parked_thread_count() == 0


# ---------------------------------------------------------------------------
# PipelineWorker.run — every way out, with no journal to write it into
# ---------------------------------------------------------------------------
#
# ``journal=False`` is the read-only housekeeping case: a history refresh, a
# model scan. It has no manifest to mark failed, so every exit branch has a
# second shape with nothing to record — and each still has to announce
# itself on ``finished`` and ``error``.

def _run_worker(fn, **kwargs):
    """Run a worker on this thread and collect everything it announced."""
    worker = B.PipelineWorker(fn, {}, journal=False, **kwargs)
    seen = {"lines": [], "errors": [], "finished": [], "results": []}
    worker.line_ready.connect(seen["lines"].append)
    worker.error.connect(seen["errors"].append)
    worker.finished.connect(seen["finished"].append)
    worker.result_ready.connect(seen["results"].append)
    worker.run()
    return worker, seen


def test_a_cancelled_unjournalled_run_still_says_it_stopped_safely():
    def _fn(_settings):
        raise PipelineCancelled("the user pressed Stop")

    worker, seen = _run_worker(_fn)

    assert worker.was_cancelled is True
    assert seen["finished"] == [False]
    assert seen["errors"] == []
    assert any("Cancelled safely: the user pressed Stop" in line
               for line in seen["lines"])


def test_a_nonzero_sys_exit_is_a_failure_even_with_no_manifest():
    """``sys.exit(1)`` used to read as a green, completed run."""
    def _fn(_settings):
        raise SystemExit(2)

    _worker, seen = _run_worker(_fn)

    assert seen["finished"] == [False]
    assert len(seen["errors"]) == 1
    assert "SystemExit" in seen["errors"][0]


def test_a_zero_sys_exit_is_an_early_success():
    def _fn(_settings):
        raise SystemExit(0)

    _worker, seen = _run_worker(_fn)

    assert seen["finished"] == [True]
    assert seen["errors"] == []


def test_an_ordinary_failure_with_no_manifest_still_reports_its_traceback():
    def _fn(_settings):
        raise ValueError("the plate folder holds no images")

    _worker, seen = _run_worker(_fn)

    assert seen["finished"] == [False]
    assert "the plate folder holds no images" in seen["errors"][0]


def test_a_keyboard_interrupt_with_no_manifest_is_reported_not_swallowed():
    """A BaseException must not leave the caller waiting for ``finished``."""
    def _fn(_settings):
        raise KeyboardInterrupt()

    _worker, seen = _run_worker(_fn)

    assert seen["finished"] == [False]
    assert "KeyboardInterrupt" in seen["errors"][0]


def test_a_result_that_cannot_be_delivered_does_not_fail_the_run(caplog):
    """The payload is a convenience; the run's verdict does not depend on it."""
    class _Undeliverable:
        def emit(self, *_args):
            raise RuntimeError("the receiving screen has been destroyed")

    worker = B.PipelineWorker(lambda _settings: {"rows": 4}, {}, journal=False)
    worker.result_ready = _Undeliverable()
    finished, errors = [], []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    with caplog.at_level(logging.DEBUG, logger=B.LOG.name):
        worker.run()

    assert finished == [True], "an undeliverable result failed the run"
    assert errors == []
    assert "could not deliver the pipeline result" in caplog.text


def test_a_result_is_handed_back_before_the_run_is_announced_as_over():
    order = []
    worker = B.PipelineWorker(lambda _settings: {"rows": 4}, {}, journal=False)
    worker.result_ready.connect(lambda payload: order.append(("result", payload)))
    worker.finished.connect(lambda ok: order.append(("finished", ok)))

    worker.run()

    assert order == [("result", {"rows": 4}), ("finished", True)]


# ---------------------------------------------------------------------------
# Figure capture inside a run
# ---------------------------------------------------------------------------

def test_a_figure_that_would_not_render_still_gets_its_tile():
    """The PNG is a fast path for the gallery, not a condition of the tile."""
    import matplotlib.pyplot as plt
    from spacr.qt.widgets import figure_queue

    plt.close("all")
    original = figure_queue.render_figure_to_png
    figure_queue.render_figure_to_png = lambda *a, **k: False
    try:
        def _fn(settings):
            settings["fig"], ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            plt.show()

        settings = {}
        worker = B.PipelineWorker(_fn, settings, journal=False)
        seen = []
        worker.figure_ready.connect(lambda fig, png: seen.append((id(fig), png)))
        worker.run()
    finally:
        figure_queue.render_figure_to_png = original
        plt.close("all")

    assert seen == [(id(settings["fig"]), "")]


def test_a_figure_whose_manager_has_gone_is_not_recreated_to_be_shown(
        monkeypatch):
    """``pyplot.figure(number)`` is a constructor as well as a lookup.

    Another thread can close a figure between reading the numbers and
    reading the figures, so capture goes through the manager: no manager,
    no figure, and certainly no new one assigned to this run.
    """
    import matplotlib.pyplot as plt
    from matplotlib._pylab_helpers import Gcf

    plt.close("all")
    monkeypatch.setattr(Gcf, "get_fig_manager", staticmethod(lambda _n: None))
    try:
        def _fn(settings):
            settings["fig"], _ax = plt.subplots()
            assert plt.get_fignums(), "the figure never reached pyplot"
            plt.show()

        settings = {}
        worker = B.PipelineWorker(_fn, settings, journal=False)
        seen = []
        worker.figure_ready.connect(lambda fig, png: seen.append(id(fig)))
        worker.run()
    finally:
        monkeypatch.undo()
        plt.close("all")

    assert seen == []


# ---------------------------------------------------------------------------
# Settings validation around an entry point
# ---------------------------------------------------------------------------

def test_a_settings_checker_that_breaks_never_stops_the_run(monkeypatch, capsys):
    """The checker only advises; a broken one must not cost anyone their run."""
    from spacr import validate

    def _explode(_settings, _app_key):
        raise RuntimeError("the settings schema could not be loaded")

    monkeypatch.setattr(validate, "validate_settings", _explode)

    ran = []
    wrapped = B._say_what_is_wrong_with_the_settings(
        "measure", lambda settings: ran.append(settings))

    wrapped({"n_job": 4})

    assert ran == [{"n_job": 4}]
    assert "[settings]" not in capsys.readouterr().out


def test_a_problem_with_no_fix_prints_the_problem_and_nothing_else(
        monkeypatch, capsys):
    from spacr import validate

    monkeypatch.setattr(
        validate, "validate_settings",
        lambda _settings, _app_key: [
            validate.Problem(validate.WARNING, "", "This plate has one row.",
                             ""),
            validate.Problem(validate.ERROR, "n_job",
                             "n_job is not a setting.", "Use n_jobs."),
        ])

    wrapped = B._say_what_is_wrong_with_the_settings(
        "measure", lambda settings: "ran")

    assert wrapped({"n_job": 4}) == "ran"

    printed = capsys.readouterr().out.splitlines()
    assert printed == [
        "[settings] WARNING: This plate has one row.",
        "[settings] ERROR [n_job]: n_job is not a setting.",
        "[settings]     Use n_jobs.",
    ]


def test_wrapping_nothing_yields_nothing():
    assert B._say_what_is_wrong_with_the_settings("measure", None) is None


def test_a_wrapped_entry_point_called_with_no_settings_is_passed_through():
    def _fn(*args, **kwargs):
        return (args, kwargs)

    wrapped = B._say_what_is_wrong_with_the_settings("measure", _fn)

    assert wrapped(None, 1, key=2) == ((1,), {"key": 2})


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def test_an_app_key_nothing_claims_resolves_to_no_entry_point():
    """Interactive-only and unknown modules both answer None, not a stub."""
    assert B.resolve_pipeline_entry("no_module_is_registered_under_this") is None


# ---------------------------------------------------------------------------
# make_thread's pre-import of pyplot
# ---------------------------------------------------------------------------

def test_the_pyplot_pre_import_leaves_the_collector_as_it_found_it(monkeypatch):
    """GC is suspended for the import only when it was running to begin with.

    The import is done here, on the caller's thread, because doing it first
    on a worker has aborted the process. Suspending an already-suspended
    collector and re-enabling it afterwards would hand the caller a
    collector it never asked for.
    """
    import builtins

    real_import = builtins.__import__
    monkeypatch.delitem(sys.modules, "matplotlib.pyplot", raising=False)

    def _no_pyplot(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("no matplotlib in this build")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_pyplot)
    monkeypatch.setattr(B, "_REGISTRY", B.RunRegistry())

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        thread, worker = B.make_thread(lambda settings: None, {},
                                       app_key="measure")
        assert gc.isenabled() is False, "the collector was switched back on"
    finally:
        if was_enabled:
            gc.enable()
        monkeypatch.undo()

    assert isinstance(thread, QThread)
    assert isinstance(worker, B.PipelineWorker)
    assert thread.isRunning() is False
