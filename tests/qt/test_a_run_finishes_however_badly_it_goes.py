"""A pipeline run announces its end no matter what fails on the way.

:class:`~spacr.qt.bridge.PipelineWorker.run` is the one place a spaCR run
happens, and everything it sets up around the pipeline function is optional
scaffolding: the matplotlib capture, the figure sink, the reproducibility
manifest, the stream redirection. None of it is the work, and every piece of
it can fail on somebody's machine -- a stripped matplotlib, a read-only runs
directory, a figure object that refuses attributes.

The invariant is that ``finished`` is emitted exactly once with an honest
verdict, the streams are handed back, and anything that went wrong with the
scaffolding arrives as a line in the console rather than as a traceback that
loses the run.

The other half of the file is the small guards around the run registry: a
worker count that is not a number, a QThread whose C++ half PySide6 has
already taken away, a callable that will not carry an attribute.
"""
from __future__ import annotations

import sys

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QThread  # noqa: E402

from spacr.qt import bridge as B  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_worker_journal(monkeypatch, tmp_path):
    """A run must not write test manifests into the user's real journal."""
    from spacr import run_journal

    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: root)
    return root


def _run(fn, settings=None, **kwargs):
    """Run one worker inline and return ``(lines, errors, finished)``."""
    worker = B.PipelineWorker(fn, dict(settings or {}), **kwargs)
    lines, errors, finished = [], [], []
    worker.line_ready.connect(lines.append)
    worker.error.connect(errors.append)
    worker.finished.connect(finished.append)
    worker.run()
    return worker, lines, errors, finished


# ---------------------------------------------------------------------------
# Stopped before it started
# ---------------------------------------------------------------------------

def test_a_run_cancelled_before_it_starts_does_no_setup_at_all(
        qtbot, _isolated_worker_journal):
    """Stop can be clicked in the tick between ``start()`` and ``run()``.

    Nothing was produced, so nothing may be set up: no manifest is opened for
    work that never happened, and the pause gate is released so the thread can
    retire immediately.
    """
    def _never_called(_settings):
        raise AssertionError("the pipeline ran after being cancelled")

    worker = B.PipelineWorker(_never_called, {}, app_key="measure")
    worker.request_cancel("stopped in the same tick")
    lines, finished = [], []
    worker.line_ready.connect(lines.append)
    worker.finished.connect(finished.append)

    worker.run()

    assert finished == [False]
    assert worker.was_cancelled is True
    assert any("Cancelled before start" in line for line in lines)
    assert any("stopped in the same tick" in line for line in lines)
    assert list(_isolated_worker_journal.iterdir()) == []
    assert worker.gate.is_paused() is False


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------

def test_a_journal_that_cannot_be_opened_is_a_warning_not_a_lost_run(
        qtbot, monkeypatch):
    """The manifest is a record OF the run, never a precondition for it."""
    from spacr import run_journal

    def _explode(*_a, **_k):
        raise OSError("the runs directory is read only")

    monkeypatch.setattr(run_journal, "open_run", _explode)
    ran = []

    _worker, lines, errors, finished = _run(
        lambda settings: ran.append(True), app_key="measure")

    assert ran == [True], "the pipeline ran anyway"
    assert finished == [True]
    assert errors == []
    assert any("could not open reproducibility manifest" in line
               for line in lines)
    assert any("the runs directory is read only" in line for line in lines)


def test_hashing_is_announced_only_when_it_is_going_to_happen(qtbot):
    """The line names a pause. Printing it with hashing off names one that is
    not there, and claims a record that was not made."""
    _w, quiet, _e, _f = _run(lambda settings: None, {"hash_inputs": False},
                             app_key="measure")
    _w2, loud, _e2, _f2 = _run(lambda settings: None, {"hash_inputs": True},
                               app_key="measure")

    assert not any("input hashes" in line for line in quiet)
    assert any("input hashes" in line for line in loud)


def test_a_journal_that_cannot_be_closed_is_a_warning_too(qtbot, monkeypatch):
    """The run already happened; failing to finalise cannot un-happen it."""
    from spacr import run_journal

    real = run_journal.open_run

    class _RefusesToClose:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            return self._inner.__enter__()

        def __exit__(self, *_exc):
            raise OSError("the manifest could not be written")

    monkeypatch.setattr(run_journal, "open_run",
                        lambda *a, **k: _RefusesToClose(real(*a, **k)))

    _worker, lines, errors, finished = _run(lambda settings: None,
                                            app_key="measure")

    assert finished == [True]
    assert errors == []
    assert any("could not finalize reproducibility manifest" in line
               for line in lines)


def test_a_cancelled_run_is_recorded_as_cancelled_not_as_a_failure(
        qtbot, _isolated_worker_journal):
    import json

    def _stops(_settings):
        raise B.PipelineCancelled("stopped at a safe boundary")

    worker, lines, errors, finished = _run(_stops, app_key="measure")

    assert finished == [False]
    assert errors == [], "a cancellation is not an error"
    assert worker.was_cancelled is True
    assert any("Cancelled safely" in line for line in lines)
    manifest = json.loads(
        next(_isolated_worker_journal.glob("*/manifest.json")).read_text())
    assert manifest["status"] == "cancelled"


def test_an_aborted_run_leaves_a_failed_manifest_not_a_running_one(
        qtbot, _isolated_worker_journal):
    """A KeyboardInterrupt is not an ``Exception``.

    Without its own arm it would leave the manifest saying "running" for ever,
    which is the one state nobody can act on.
    """
    import json

    def _aborts(_settings):
        raise KeyboardInterrupt()

    _worker, _lines, errors, finished = _run(_aborts, app_key="measure")

    assert finished == [False]
    assert len(errors) == 1
    assert "KeyboardInterrupt" in errors[0]
    manifest = json.loads(
        next(_isolated_worker_journal.glob("*/manifest.json")).read_text())
    assert manifest["status"] == "failed"


# ---------------------------------------------------------------------------
# The figure capture
# ---------------------------------------------------------------------------

def test_a_qt_backend_is_forced_to_agg_before_the_run_draws_anything(
        qtbot, monkeypatch):
    """A figure built on a Qt backend inside a worker thread carries a
    QWidget owned by that thread, which aborts the process on teardown."""
    import matplotlib

    used = []
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "QtAgg")
    monkeypatch.setattr(matplotlib, "use",
                        lambda name, force=False: used.append((name, force)))

    _run(lambda settings: None)

    assert used == [("Agg", True)]


def test_a_figure_that_refuses_a_marker_is_still_delivered(qtbot, monkeypatch):
    """The marker is half of a double-check, not the delivery itself."""
    import matplotlib
    matplotlib.get_backend()

    class _NoAttributes:
        __slots__ = ()

    figure = _NoAttributes()
    seen = []

    def _draws(_settings):
        from spacr import figure_sink
        figure_sink.publish(figure)

    worker = B.PipelineWorker(_draws, {})
    worker.figure_ready.connect(lambda fig, png: seen.append((fig, png)))
    worker.run()

    assert [fig for fig, _png in seen] == [figure]


def test_a_figure_that_cannot_be_rendered_is_still_delivered(
        qtbot, monkeypatch):
    """The prerendered PNG is a convenience for the tile; the figure is the
    answer, and a run must not lose it because a thumbnail failed."""
    from spacr.qt.widgets import figure_queue

    def _explode(*_a, **_k):
        raise RuntimeError("no writable temp directory")

    monkeypatch.setattr(figure_queue, "render_figure_to_png", _explode)

    from matplotlib.figure import Figure
    figure = Figure()
    seen = []

    def _draws(_settings):
        from spacr import figure_sink
        figure_sink.publish(figure)

    worker = B.PipelineWorker(_draws, {})
    worker.figure_ready.connect(lambda fig, png: seen.append((fig, png)))
    worker.run()

    assert seen == [(figure, "")]


def test_a_figure_sink_that_cannot_be_installed_does_not_stop_the_run(
        qtbot, monkeypatch):
    from spacr import figure_sink

    def _explode(_fn):
        raise RuntimeError("the sink module is broken")

    monkeypatch.setattr(figure_sink, "set_sink", _explode)
    ran = []

    _worker, _lines, errors, finished = _run(
        lambda settings: ran.append(True))

    assert ran == [True]
    assert finished == [True]
    assert errors == []


def test_a_sink_that_cannot_be_cleared_does_not_swallow_the_verdict(
        qtbot, monkeypatch):
    from spacr import figure_sink

    def _explode():
        raise RuntimeError("the sink module is broken")

    monkeypatch.setattr(figure_sink, "clear_sink", _explode)

    _worker, _lines, errors, finished = _run(lambda settings: None)

    assert finished == [True]
    assert errors == []


def test_streams_are_handed_back_even_when_the_flush_fails(
        qtbot, monkeypatch):
    """The redirector is torn down in ``finally``; a flush that raises there
    would take the ``finished`` emit with it."""
    stdout_before, stderr_before = sys.stdout, sys.stderr

    def _explode(self):
        raise RuntimeError("the console has gone")

    monkeypatch.setattr(B._StreamRedirector, "flush", _explode)

    _worker, _lines, _errors, finished = _run(lambda settings: None)

    assert finished == [True]
    assert sys.stdout is stdout_before or isinstance(
        sys.stdout, B._ThreadStreamRouter)
    assert sys.stderr is stderr_before or isinstance(
        sys.stderr, B._ThreadStreamRouter)


# ---------------------------------------------------------------------------
# The stream plumbing
# ---------------------------------------------------------------------------

def test_the_redirector_accepts_what_a_pipeline_actually_writes():
    """``print`` is not the only writer: a C extension can hand bytes-like
    objects to ``sys.stdout.write``, and a redirector that assumed str
    concatenated its way to a TypeError inside somebody's run."""
    received = []
    redirect = B._StreamRedirector(received.append)

    redirect.write(12345)
    redirect.write("\n")

    assert received == ["12345\n"]


def test_a_console_that_raises_does_not_break_the_pipeline_writing_to_it():
    """The consumer is a GUI widget; the producer is somebody's analysis."""
    def _explode(_text):
        raise RuntimeError("the console widget has been deleted")

    redirect = B._StreamRedirector(_explode)

    redirect.write("a line\n")   # must not raise
    redirect.flush()


def test_the_router_survives_a_target_that_cannot_flush():
    router = B._ThreadStreamRouter(sys.__stdout__)

    class _Unflushable:
        def flush(self):
            raise RuntimeError("gone")

        def write(self, text):
            return len(text)

    target = _Unflushable()
    router.register(target)
    try:
        router.flush()          # must not raise
    finally:
        router.unregister(target)


def test_the_router_reports_the_original_streams_terminal_and_encoding():
    router = B._ThreadStreamRouter(sys.__stdout__)

    assert router.encoding == getattr(sys.__stdout__, "encoding", None)
    assert router.isatty() == bool(sys.__stdout__.isatty())


def test_a_second_worker_reuses_the_routers_already_installed(monkeypatch):
    """Installing a second pair would orphan the first and lose its threads."""
    first = B._StreamRedirector(lambda _t: None)
    second = B._StreamRedirector(lambda _t: None)

    out1, err1 = B._register_worker_streams(first)
    try:
        out2, err2 = B._register_worker_streams(second)
        try:
            assert out2 is out1
            assert err2 is err1
        finally:
            B._unregister_worker_streams(second, out2, err2)
    finally:
        B._unregister_worker_streams(first, out1, err1)


# ---------------------------------------------------------------------------
# Worker budgets
# ---------------------------------------------------------------------------

def test_a_capacity_that_is_not_a_number_is_one_worker():
    """Better one worker than a traceback out of the allocator."""
    assert B.worker_capacity("as many as you like") == 1
    assert B.worker_capacity(None) >= 1


def test_a_worker_setting_that_is_not_a_number_is_left_alone():
    """Rewriting it to a guess would hide a typo in a settings CSV."""
    settings = {"n_jobs": "lots"}

    B.apply_worker_budget(settings, total=8)

    assert settings["n_jobs"] == "lots"


# ---------------------------------------------------------------------------
# Threads that are already gone
# ---------------------------------------------------------------------------

def _deleted_thread():
    """A QThread whose C++ half has been taken away."""
    import shiboken6

    thread = QThread()
    shiboken6.delete(thread)
    return thread


def test_a_thread_with_no_c_plus_plus_half_is_not_running():
    assert B.thread_has_stopped(None) is True
    assert B.thread_has_stopped(_deleted_thread()) is True


def _handle():
    worker = B.PipelineWorker(lambda settings: None, {})
    return B.RunHandle("measure", worker, QThread())


def test_a_handle_whose_thread_is_gone_reports_itself_stopped():
    handle = _handle()

    handle.thread = None
    assert handle.is_running() is False

    handle.thread = _deleted_thread()
    assert handle.is_running() is False


def test_cancelling_a_job_whose_thread_is_gone_is_not_an_error():
    handle = _handle()
    handle.thread = _deleted_thread()

    handle.request_cancel("stopped")   # must not raise

    assert handle.worker.cancel_token.cancelled


# ---------------------------------------------------------------------------
# Naming a job
# ---------------------------------------------------------------------------

def test_a_callable_that_will_not_carry_a_name_is_left_unnamed():
    """A builtin rejects attributes; the job simply goes unnamed rather than
    the whole dispatch table failing to build."""
    assert B._tag("measure", None) is None
    assert B._tag("measure", len) is len


# ---------------------------------------------------------------------------
# Pausing
# ---------------------------------------------------------------------------

def test_a_gate_reports_how_long_it_has_held_the_worker():
    """Zero when running: a pause that never happened has no duration."""
    gate = B.PauseGate()

    assert gate.paused_for() == 0.0

    gate.pause()
    assert gate.paused_for() >= 0.0
    assert gate.is_paused() is True

    gate.resume()
    assert gate.paused_for() == 0.0


def test_a_checkpoint_waits_on_the_gate_of_its_own_thread():
    """Outside a worker there is no gate, and the call is a no-op."""
    assert B.current_gate() is None
    B.checkpoint()

    gate = B.PauseGate()
    waited = []
    gate.wait_if_paused = lambda timeout=None: waited.append(True) or True
    B._LOCAL.gate = gate
    try:
        B.checkpoint()
    finally:
        del B._LOCAL.gate

    assert waited == [True]


def test_an_entry_point_that_will_not_carry_the_pausable_mark_is_returned():
    """Marking a builtin fails; shipping a Pause button that lies would be
    worse than the mark quietly not sticking."""
    assert B.pausable(len) is len
    assert B.current_gate() is None


# ---------------------------------------------------------------------------
# Threads that will not stop
# ---------------------------------------------------------------------------

class _VanishingThread:
    """A QThread whose C++ half goes away between two calls."""

    def __init__(self, running=True):
        self._running = running

    def isRunning(self):                          # noqa: N802 - Qt name
        return self._running

    def requestInterruption(self):                # noqa: N802 - Qt name
        raise RuntimeError("Internal C++ object already deleted.")

    def quit(self):
        raise RuntimeError("Internal C++ object already deleted.")

    def wait(self, _ms=0):
        raise RuntimeError("Internal C++ object already deleted.")


def test_a_thread_that_vanishes_while_being_stopped_counts_as_stopped():
    assert B.drain_thread(_VanishingThread(), None, timeout_ms=1) is True
    assert B.drain_thread(None, None, timeout_ms=1) is True


def test_a_parked_thread_that_has_gone_is_dropped_from_the_park():
    """The park exists so nothing drops the last reference to a *running*
    QThread. One that no longer exists cannot be that."""
    import shiboken6

    thread = QThread()
    with B._PARKED_LOCK:
        before = list(B._PARKED_THREADS)
        B._PARKED_THREADS.append((thread, None))
    try:
        shiboken6.delete(thread)
        assert B.prune_parked_threads() == len(before)
        assert B.parked_thread_count() == len(before)
    finally:
        with B._PARKED_LOCK:
            B._PARKED_THREADS[:] = before


def test_stopping_every_job_gives_up_when_the_budget_is_spent():
    """The shutdown budget is a budget: a job that will not stop is reported,
    not waited on for ever."""
    waited = []

    class _Stubborn(_VanishingThread):
        def wait(self, ms=0):
            waited.append(ms)
            return False

    worker = B.PipelineWorker(lambda settings: None, {})
    registry = B.RunRegistry()
    handle = B.RunHandle("measure", worker, _Stubborn())
    registry.register(handle)

    left = registry.cancel_all(timeout_ms=0)

    assert waited == [], "a spent budget is not spent again"
    assert [h.app_key for h in left] == ["measure"]
    assert worker.cancel_token.cancelled


def test_a_job_whose_thread_vanishes_while_shutting_down_is_not_an_error():
    worker = B.PipelineWorker(lambda settings: None, {})
    registry = B.RunRegistry()
    registry.register(B.RunHandle("measure", worker, _VanishingThread()))

    registry.cancel_all(timeout_ms=50)   # must not raise


# ---------------------------------------------------------------------------
# Naming and checking settings
# ---------------------------------------------------------------------------

def test_an_app_with_no_entry_point_is_not_wrapped():
    assert B._say_what_is_wrong_with_the_settings("measure", None) is None


def test_an_entry_point_called_with_no_settings_is_passed_none_at_all():
    """Some entry points take no argument. Handing them ``None`` would be a
    TypeError blamed on the pipeline rather than on the wrapper."""
    seen = []
    wrapped = B._say_what_is_wrong_with_the_settings(
        "measure", lambda *args: seen.append(args))

    wrapped()

    assert seen == [()]


def test_a_plugin_entry_that_is_not_callable_is_refused_and_logged(
        monkeypatch, caplog):
    """``load_object`` can return a module or a string; running one would be
    a TypeError from inside the worker thread instead of a refusal here."""
    import logging
    import types

    from spacr import plugins

    app = types.SimpleNamespace(entrypoint="somewhere:not_a_function")
    monkeypatch.setattr(plugins, "get_app",
                        lambda key: app if key == "a_plugin" else None)
    monkeypatch.setattr(plugins, "load_object", lambda _path: "not callable")

    with caplog.at_level(logging.ERROR, logger="spacr.qt.bridge"):
        assert B.resolve_pipeline_entry("a_plugin") is None

    assert any("Could not resolve pipeline entry" in record.getMessage()
               for record in caplog.records)


# ---------------------------------------------------------------------------
# plt.show() inside a run
# ---------------------------------------------------------------------------

def test_a_figure_shown_by_the_pipeline_reaches_the_gallery(qtbot):
    """``plt.show()`` in a worker would open a blocking window; the run
    captures it and hands the figure to the UI instead."""
    seen = []

    def _draws(_settings):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot([0, 1], [0, 1])
        plt.show()

    worker = B.PipelineWorker(_draws, {})
    worker.figure_ready.connect(lambda fig, png: seen.append((fig, png)))
    try:
        worker.run()
    finally:
        import matplotlib.pyplot as plt
        plt.close("all")

    assert len(seen) == 1
    assert seen[0][1], "the figure was prerendered for the tile"


def test_a_figure_open_before_the_run_is_not_claimed_by_the_gallery(qtbot):
    """A worker publishes only figures made by that run.

    pyplot's registry is process-global, so an unrelated screen may own an
    open figure when the run starts.  Figure numbers are reusable; object
    identity is the ownership boundary.
    """
    import matplotlib.pyplot as plt

    stale = plt.figure()
    seen = []

    def _draws(_settings):
        fresh = plt.figure()
        plt.show()
        assert fresh is not stale

    worker = B.PipelineWorker(_draws, {})
    worker.figure_ready.connect(lambda fig, png: seen.append((fig, png)))
    try:
        worker.run()
    finally:
        plt.close("all")

    assert len(seen) == 1
    assert seen[0][0] is not stale


def test_a_shown_figure_that_cannot_be_prerendered_is_still_handed_over(
        qtbot, monkeypatch):
    from spacr.qt.widgets import figure_queue

    def _explode(*_a, **_k):
        raise RuntimeError("no writable temp directory")

    monkeypatch.setattr(figure_queue, "render_figure_to_png", _explode)
    seen = []

    def _draws(_settings):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.show()

    worker = B.PipelineWorker(_draws, {})
    worker.figure_ready.connect(lambda fig, png: seen.append((fig, png)))
    try:
        worker.run()
    finally:
        import matplotlib.pyplot as plt
        plt.close("all")

    assert len(seen) == 1
    assert seen[0][1] == ""


def test_a_run_without_matplotlib_at_all_still_finishes(qtbot, monkeypatch):
    """A headless build with no matplotlib runs the pipeline and captures
    nothing, rather than failing before the work starts."""
    import builtins

    real_import = builtins.__import__

    def _no_matplotlib(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("no matplotlib in this build")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_matplotlib)
    ran = []

    _worker, _lines, errors, finished = _run(lambda settings: ran.append(True))

    assert ran == [True]
    assert finished == [True]
    assert errors == []


def test_the_show_router_is_restored_even_when_removing_it_fails(
        qtbot, monkeypatch):
    """Teardown runs in ``finally``; anything raising there would take the
    ``finished`` emit with it."""
    def _explode(_target):
        raise RuntimeError("the router registry is confused")

    monkeypatch.setattr(B, "_unregister_matplotlib_show", _explode)

    _worker, _lines, errors, finished = _run(lambda settings: None)

    assert finished == [True]
    assert errors == []


def test_a_gate_the_pipeline_removed_is_not_removed_twice(qtbot):
    """The gate is thread-local and the pipeline can reach it; a run that
    tidied it away must still finish."""
    def _drops_the_gate(_settings):
        del B._LOCAL.gate

    _worker, _lines, errors, finished = _run(_drops_the_gate)

    assert finished == [True]
    assert errors == []
    assert B.current_gate() is None


def test_a_handle_whose_worker_has_been_deleted_still_retires():
    """``retire`` is wired to ``QThread.finished``; on shutdown that can
    arrive after PySide6 has already taken the worker's C++ half away."""
    import shiboken6

    worker = B.PipelineWorker(lambda settings: None, {})
    handle = B.RunHandle("measure", worker, QThread())
    shiboken6.delete(worker)

    handle.retire()

    assert handle.worker is None
    assert handle.thread is None


def test_a_job_can_be_built_without_matplotlib_being_importable(monkeypatch):
    """``make_thread`` pre-imports pyplot on the CALLER's thread on purpose.

    The first ``import matplotlib.pyplot`` inside a QThread has aborted the
    process, so it is done here where it is safe. A build that has no
    matplotlib still has to be able to start a job -- the worker's own import
    is what would fail, and that path already handles it.
    """
    import builtins

    real_import = builtins.__import__
    monkeypatch.delitem(sys.modules, "matplotlib.pyplot", raising=False)

    def _no_pyplot(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("no matplotlib in this build")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_pyplot)

    thread, worker = B.make_thread(lambda settings: None, {},
                                   app_key="measure")
    try:
        assert isinstance(thread, QThread)
        assert isinstance(worker, B.PipelineWorker)
        assert thread.isRunning() is False
    finally:
        monkeypatch.undo()
        for handle in list(B.registry().active()):
            if handle.worker is worker:
                handle.retire()
