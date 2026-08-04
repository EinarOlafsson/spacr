"""Teardown contract for every Qt worker thread in the application.

Written from two pre-existing suite failures — ``tests/qt/test_onboarding``
live-locking after certain other Qt files, and a 30-file Qt shard dumping
core with ``exit=139`` — which turned out to be one family: work that keeps
running after the thing that owned it stopped owning it. Each defect found
on the way is pinned here.

* **The console log sink re-entered itself.** Writing a line runs Python
  inside a QWidget; with verbose logging on, ``spacr.logging_util``'s
  profile hook logs on entry to every spaCR function it passes through, and
  both console sinks feed that record straight back into the same widget.
  ``_StdoutBlock.append`` answers it with a nested ``setPlainText``, whose
  first act is to destroy the QTextDocument's frames — the ones the outer
  call is still inside. gdb: ``QTextFrame::~QTextFrame ->
  QTextDocumentPrivate::clear -> QTextDocument::setPlainText``, ``#0`` in
  freed memory. Reproduced as ``pytest tests/qt/test_all_module_smoke.py
  tests/qt/test_batch_f_diagnostics.py`` (exit 139); with the guard removed
  the same loop spins instead of crashing.
* ``spacr.qt.ai.worker.make_stream_thread`` queued ``worker.finished ->
  thread.quit`` to the **GUI-affine** QThread object, so a GUI thread that
  went straight into ``thread.wait()`` (which is what every "drain before
  closing" path does) waited out its whole timeout on a worker that had
  already finished — and then reached ``QThread.terminate()``.
* the same function scheduled the worker for C++ deletion while Python
  still owned it, the exact double-ownership segfault
  :func:`spacr.qt.bridge.make_thread`'s ownership essay records.
* a read-only background job that outlived the screen which started it
  stayed in the process-wide run registry, where ``MainWindow.closeEvent``
  read it as "analysis still running", refused to close and put up a modal.
  On a headless run nothing can dismiss that modal, so the process spins in
  its nested event loop forever — captured live, with the GUI thread parked
  in ``spacr/qt/app.py`` ``closeEvent``.
* three screens retired their worker pairs with an ``isRunning()`` filter
  on ``thread.finished``, which raises out of the slot once ``deleteLater``
  has reaped the QThread and therefore retired nothing at all.
"""
from __future__ import annotations

import os
import threading
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, QThread, Qt, Signal          # noqa: E402
from PySide6.QtWidgets import QWidget                            # noqa: E402

from spacr.cancellation import checkpoint as cancellation_checkpoint  # noqa
from spacr.qt.bridge import (                                    # noqa: E402
    drain_thread, make_thread, parked_thread_count,
    prune_parked_threads, registry,
)


@pytest.fixture(autouse=True)
def _isolated_run_journal(monkeypatch, tmp_path):
    """Never write into the user's real ``~/.spacr/runs`` from a test."""
    from spacr import run_journal

    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: root)
    return root


def _retired(thread) -> bool:
    """Has ``thread`` stopped — including "its C++ half is already gone"?

    The GUI pump that lets a job finish also delivers ``thread.finished``
    and therefore runs ``thread.deleteLater``, so polling the same Python
    wrapper afterwards raises ``RuntimeError`` rather than returning False.
    A reaped wrapper is the strongest possible evidence of retirement:
    ``deleteLater`` is only ever posted from ``thread.finished``.
    """
    try:
        return not thread.isRunning()
    except RuntimeError:
        return True


def _join(qtbot, thread, timeout_ms: int = 5000) -> None:
    """Wait for ``thread`` to retire while pumping the GUI event loop."""
    qtbot.waitUntil(lambda t=thread: _retired(t), timeout=timeout_ms)


# ---------------------------------------------------------------------------
# 1. A worker's completion handler must run on the GUI thread
# ---------------------------------------------------------------------------

class _Host(QWidget):
    """A GUI-affine receiver that records where its handler ran."""

    def __init__(self) -> None:
        super().__init__()
        self.handler_thread = None
        self.line_thread = None
        self.error_thread = None

    def on_finished(self, _ok: bool) -> None:
        self.handler_thread = QThread.currentThread()

    def on_line(self, _text: str) -> None:
        self.line_thread = QThread.currentThread()

    def on_error(self, _text: str) -> None:
        self.error_thread = QThread.currentThread()


def test_pipeline_worker_signals_are_handled_on_the_gui_thread(
        qtbot, qt_theme_applied):
    """Every screen wires these three; all three must land on the GUI thread.

    ``PipelineWorker`` is moved to the worker thread, so PySide6 picks the
    connection type from *its* affinity: a handler that is not a bound
    method of a GUI-affine QObject gets a DirectConnection and runs on the
    worker thread, where every widget call inside it is undefined
    behaviour. This test fails the moment any of the three is rewired that
    way — the failure message names which one.
    """
    host = _Host()
    qtbot.addWidget(host)
    gui = host.thread()

    def _job(_settings):
        print("a line of output\n")
        raise RuntimeError("deliberate")

    thread, worker = make_thread(
        _job, {}, app_key="gui-affinity-audit", journal=False)
    worker.line_ready.connect(host.on_line)
    worker.error.connect(host.on_error)
    worker.finished.connect(host.on_finished)
    thread.start()
    _join(qtbot, thread)
    qtbot.waitUntil(lambda: host.handler_thread is not None, timeout=5000)
    qtbot.waitUntil(lambda: host.error_thread is not None, timeout=5000)

    assert host.handler_thread is gui, (
        "worker.finished handler ran on the worker thread; connect it to a "
        "bound method of a GUI-affine QObject (or relay it through one of "
        "that object's Signals) instead of a closure")
    assert host.error_thread is gui, "worker.error handler ran off the GUI thread"
    assert host.line_thread is gui, "worker.line_ready handler ran off the GUI thread"


def test_a_closure_really_would_have_run_on_the_worker_thread(
        qtbot, qt_theme_applied):
    """The check above has teeth only if the bad wiring is genuinely bad.

    Measured rather than asserted from the documentation: connect a plain
    closure to the same signal and record where it runs. If PySide6 ever
    changes this, the test above stops being a guard and this one says so.
    """
    gui = QThread.currentThread()
    seen = {}

    thread, worker = make_thread(
        lambda _s: None, {}, app_key="closure-affinity-audit", journal=False)
    worker.finished.connect(
        lambda _ok: seen.setdefault("thread", QThread.currentThread()))
    thread.start()
    _join(qtbot, thread)

    assert seen.get("thread") is not None
    assert seen["thread"] is not gui, (
        "a closure on worker.finished no longer runs on the worker thread; "
        "the GUI-affinity guard above is measuring nothing")


# ---------------------------------------------------------------------------
# 2. The thread census returns to baseline — completed AND cancelled
# ---------------------------------------------------------------------------

def _os_threads() -> int:
    """Threads the OS sees, native ones included (Linux), else Python's view."""
    try:
        return len(os.listdir("/proc/self/task"))
    except OSError:
        return threading.active_count()


#: A per-run leak shows up as one thread per cycle, i.e. 20 over the loops
#: below. Qt, OpenMP and torch each start a small *fixed* number of native
#: helpers the first time certain code paths run, and a warm-up cycle cannot
#: be relied on to have touched all of them. Four is comfortably above that
#: fixed noise and far below the smallest real leak.
_SLACK = 4
_CYCLES = 20


def _drained(prefix: str) -> bool:
    return not any(
        h.app_key.startswith(prefix) for h in registry().active())


def test_thread_census_returns_to_baseline_after_runs_complete(
        qtbot, qt_theme_applied):
    """Twenty jobs run to completion must not grow the process."""
    def _cycle(index: int):
        ran = threading.Event()
        thread, worker = make_thread(
            lambda _s: ran.set(), {},
            app_key=f"census-done-{index}", journal=False)
        thread.start()
        _join(qtbot, thread)
        assert ran.is_set(), f"cycle {index} never entered the pipeline body"
        return worker

    _cycle(-1)          # warm-up: first job imports matplotlib + the journal
    qtbot.waitUntil(lambda: _drained("census-done-"), timeout=5000)
    base_py, base_os = threading.active_count(), _os_threads()

    for index in range(_CYCLES):
        assert not _cycle(index).was_cancelled

    qtbot.waitUntil(lambda: _drained("census-done-"), timeout=5000)
    assert threading.active_count() - base_py <= 0, (
        f"Python threads grew {base_py} -> {threading.active_count()} "
        f"over {_CYCLES} completed runs")
    assert _os_threads() - base_os <= _SLACK, (
        f"OS threads grew {base_os} -> {_os_threads()} over {_CYCLES} "
        f"completed runs; a per-run leak would show up as ~{_CYCLES}")


def test_thread_census_returns_to_baseline_after_runs_are_cancelled(
        qtbot, qt_theme_applied):
    """…and twenty jobs stopped mid-flight must not either.

    A cancelled run takes a different exit from ``PipelineWorker.run``
    (``PipelineCancelled`` unwinds through the ``finally``), so "completed
    runs do not leak" says nothing about it. This is the path a user takes
    every time they press Stop.
    """
    def _cycle(index: int):
        started = threading.Event()

        def _job(_settings):
            started.set()
            while True:
                cancellation_checkpoint()
                time.sleep(0.002)

        thread, worker = make_thread(
            _job, {}, app_key=f"census-stop-{index}", journal=False)
        thread.start()
        assert started.wait(5), f"cycle {index} never started"
        worker.request_cancel("census stop")
        thread.requestInterruption()
        _join(qtbot, thread)
        assert worker.was_cancelled, f"cycle {index} did not record a cancel"
        return worker

    _cycle(-1)
    qtbot.waitUntil(lambda: _drained("census-stop-"), timeout=5000)
    base_py, base_os = threading.active_count(), _os_threads()

    for index in range(_CYCLES):
        _cycle(index)

    qtbot.waitUntil(lambda: _drained("census-stop-"), timeout=5000)
    assert threading.active_count() - base_py <= 0, (
        f"Python threads grew {base_py} -> {threading.active_count()} "
        f"over {_CYCLES} cancelled runs")
    assert _os_threads() - base_os <= _SLACK, (
        f"OS threads grew {base_os} -> {_os_threads()} over {_CYCLES} "
        f"cancelled runs; a per-run leak would show up as ~{_CYCLES}")


# ---------------------------------------------------------------------------
# 3. quit() must reach the worker's own loop, not the GUI thread's queue
# ---------------------------------------------------------------------------

class _TinyWorker(QObject):
    finished = Signal(bool, str)

    def run(self) -> None:
        self.finished.emit(True, "done")


def _wait_from_a_blocked_gui_thread(connection) -> bool:
    """Wire ``finished -> quit`` with ``connection`` and wait without pumping.

    Returns whether ``thread.wait()`` saw the thread stop. This is exactly
    what ``ConsolePanel.shutdown`` does, and the only thing that differs
    between the shipped code and the bug is the connection type.
    """
    owner = QWidget()
    thread = QThread(owner)
    worker = _TinyWorker()
    worker.setParent(None)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit, connection)
    thread.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not thread.isFinished():
        # No processEvents(): the GUI thread is *blocked*, which is the
        # whole point. A queued quit() cannot be delivered from here.
        if thread.wait(200):
            break
    stopped = thread.wait(1500)
    if not stopped:
        thread.quit()
        thread.wait(3000)
    return bool(stopped)


def test_a_queued_quit_cannot_stop_a_thread_from_a_blocked_gui_thread(
        qtbot, qt_theme_applied):
    """The mechanism, measured, so the fix below is not a superstition."""
    assert _wait_from_a_blocked_gui_thread(Qt.QueuedConnection) is False
    assert _wait_from_a_blocked_gui_thread(Qt.DirectConnection) is True


def test_stream_thread_quit_is_direct_so_shutdown_never_times_out(
        qtbot, qt_theme_applied, monkeypatch):
    """``make_stream_thread`` must stop its thread from a blocked GUI thread."""
    from spacr.qt.ai import worker as worker_mod

    class _Provider:
        def stream_chat(self, _messages, system="", model=None):
            yield "hello"

        def cancel_stream(self):
            pass

    owner = QWidget()
    qtbot.addWidget(owner)
    thread, worker = worker_mod.make_stream_thread(
        _Provider(), [{"role": "user", "content": "hi"}], parent=owner)
    thread.start()
    # Block the GUI thread the way every shutdown path does. With the old
    # queued quit this wait ran its full course and the caller then reached
    # QThread.terminate().
    assert thread.wait(5000) is True, (
        "make_stream_thread's worker.finished -> thread.quit must be a "
        "DirectConnection: the QThread object is GUI-affine, so a queued "
        "quit() is posted behind this very wait()")
    assert worker is not None


def test_stream_worker_is_not_scheduled_for_cpp_deletion(
        qtbot, qt_theme_applied):
    """No ``deleteLater`` on a worker whose affinity is the worker thread.

    ``bridge.make_thread``'s ownership essay records the gdb trace and the
    measurement (3 crashes in 8 runs) for exactly this construct. The
    panels hold the worker in ``_retired`` and Python frees it there.
    """
    from spacr.qt.ai import worker as worker_mod

    class _Provider:
        def stream_chat(self, _messages, system="", model=None):
            yield "x"

        def cancel_stream(self):
            pass

    from shiboken6 import isValid

    owner = QWidget()
    qtbot.addWidget(owner)
    thread, worker = worker_mod.make_stream_thread(
        _Provider(), [], parent=owner)
    thread.start()
    assert thread.wait(5000)
    # Flush the deferred-delete queue the way a live GUI loop does. If the
    # worker had been scheduled for C++ deletion, it is gone by now; because
    # it was not, Python is still its only owner and the object is intact.
    for _ in range(5):
        qtbot.wait(20)
    assert isValid(worker), (
        "make_stream_thread scheduled the worker for C++ deletion while "
        "Python still owns it — the double-ownership segfault "
        "bridge.make_thread documents")
    assert worker.parent() is None


# ---------------------------------------------------------------------------
# 4. Nothing may terminate a thread that is running Python
# ---------------------------------------------------------------------------

def test_no_shipped_code_calls_qthread_terminate():
    """``terminate()`` is ``pthread_cancel``; a Python thread cannot survive it.

    Killed holding the GIL, the whole process stops making progress with
    every thread still alive; killed inside Qt or PySide, the heap is
    corrupt and the crash lands somewhere unrelated later. Both were live
    symptoms. ``bridge.drain_thread`` parks a stubborn thread instead.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "spacr" / "qt"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source, filename=str(path))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute)
                    and func.attr == "terminate"):
                continue
            receiver = ast.unparse(func.value).lower()
            # subprocess.Popen.terminate() is a different call and is fine.
            if "proc" in receiver or "process" in receiver:
                continue
            offenders.append(f"{path}:{node.lineno}: {ast.unparse(node)}")
    assert not offenders, (
        "QThread.terminate() is never safe here — use "
        "spacr.qt.bridge.drain_thread:\n" + "\n".join(offenders))


def test_drain_thread_stops_a_cooperative_thread_and_parks_a_stubborn_one(
        qtbot, qt_theme_applied):
    prune_parked_threads()
    before = parked_thread_count()

    release = threading.Event()

    class _Blocker(QThread):
        def run(self):
            release.wait(30)

    stubborn = _Blocker()
    stubborn.start()
    qtbot.waitUntil(lambda: stubborn.isRunning(), timeout=3000)
    # It will not stop, so drain_thread must say so rather than kill it.
    assert drain_thread(stubborn, timeout_ms=250) is False
    assert parked_thread_count() == before + 1
    assert stubborn.isRunning(), "drain_thread must never terminate a thread"

    release.set()
    qtbot.waitUntil(lambda: not stubborn.isRunning(), timeout=5000)
    assert prune_parked_threads() == before

    # A thread that stops is drained, not parked.
    quick = _Blocker()
    release2 = threading.Event()
    release2.set()
    quick.start()
    assert drain_thread(quick, timeout_ms=5000) is True
    assert parked_thread_count() == before


def test_drain_thread_tolerates_none_and_a_reaped_wrapper():
    assert drain_thread(None) is True

    class _Reaped:
        def isRunning(self):
            raise RuntimeError("Internal C++ object already deleted.")

    assert drain_thread(_Reaped()) is True


# ---------------------------------------------------------------------------
# 5. A read-only housekeeping job must never veto application shutdown
# ---------------------------------------------------------------------------

def test_housekeeping_jobs_do_not_block_shutdown(qtbot, qt_theme_applied):
    """The live-lock, at its source.

    ``MainWindow.closeEvent`` answers a non-empty ``cancel_all`` by
    refusing to close *and showing a modal*. Headless — CI, a test shard —
    nothing can dismiss that modal and the process spins in its nested
    event loop with no forward progress. A run-history refresh that
    outlived its screen used to be enough to trigger it.
    """
    block = threading.Event()
    started = threading.Event()

    def _housekeeping(_settings):
        started.set()
        block.wait(30)

    thread, worker = make_thread(
        _housekeeping, {}, app_key="history_refresh", journal=False)
    thread.start()
    try:
        assert started.wait(5)
        assert worker.blocks_shutdown is False
        remaining = registry().cancel_all(
            timeout_ms=200, reason="test shutdown")
        assert remaining == [], (
            "a read-only background job vetoed shutdown; it writes nothing, "
            "so there is no half-written artefact to protect")
    finally:
        block.set()
        _join(qtbot, thread)


def test_an_analysis_run_still_blocks_shutdown(qtbot, qt_theme_applied):
    """The other half: a run that writes outputs must still veto closing."""
    block = threading.Event()
    started = threading.Event()

    def _analysis(_settings):
        started.set()
        block.wait(30)

    thread, worker = make_thread(
        _analysis, {}, app_key="measure", journal=True)
    thread.start()
    try:
        assert started.wait(5)
        assert worker.blocks_shutdown is True
        remaining = registry().cancel_all(
            timeout_ms=200, reason="test shutdown")
        assert [h.app_key for h in remaining] == ["measure"]
    finally:
        block.set()
        _join(qtbot, thread)


# ---------------------------------------------------------------------------
# 6. The console log sink must not re-enter itself
# ---------------------------------------------------------------------------

@pytest.fixture
def _restore_console_target():
    """Put the process-wide verbose-logger console target back afterwards."""
    from spacr.qt import verbose_logger as vl

    saved = vl._console_ref
    yield vl
    vl._console_ref = saved


def test_a_record_logged_while_writing_the_console_is_dropped(
        qtbot, qt_theme_applied, _restore_console_target):
    """The loop that segfaulted the Qt shard, reduced to its mechanism.

    Writing a line into the console runs Python inside a QWidget. Any log
    record that code produces comes straight back to the same handler, and
    ``_StdoutBlock.append`` answers it with another ``setPlainText`` —
    which begins by destroying the QTextDocument's frames while the outer
    call is still inside ``QTextDocumentPrivate::clear()``. gdb put ``#0``
    in freed memory under ``QTextFrame::~QTextFrame``.
    """
    import logging

    from spacr.qt.widgets.console_panel import ConsolePanel, _StdoutBlock

    vl = _restore_console_target
    panel = ConsolePanel()
    qtbot.addWidget(panel)
    vl.register_console_target(panel)

    depth = {"now": 0, "max": 0}
    seen = []
    original = _StdoutBlock.append

    def _loud_append(self, text):
        """Log from inside the widget write — what the profile hook does.

        ``_StdoutBlock.append`` does not log on its own; it is logged
        *about*, on entry, by ``spacr.logging_util``'s function-trace hook,
        which fires for every spaCR function while verbose logging is on.
        Stating it explicitly makes the loop reproducible without depending
        on the hook's own internals.
        """
        depth["now"] += 1
        depth["max"] = max(depth["max"], depth["now"])
        seen.append(text)
        try:
            logging.getLogger("spacr.trace").debug("→ inside append")
            return original(self, text)
        finally:
            depth["now"] -= 1

    _StdoutBlock.append = _loud_append
    vl.apply_verbose_logging(True)
    try:
        logging.getLogger("spacr.trace").debug("first line")
        for _ in range(3):
            qtbot.wait(10)
    finally:
        vl.apply_verbose_logging(False)
        _StdoutBlock.append = original

    assert seen, "the console target received nothing at all"
    assert depth["max"] == 1, (
        f"the console write re-entered itself {depth['max']} deep; a record "
        "produced by a console write must be dropped, not delivered")


def test_the_console_sink_reopens_after_a_delivery(
        qtbot, qt_theme_applied, _restore_console_target):
    """The latch must be per-delivery, not a one-way switch.

    Dropping re-entrant records is only correct if ordinary ones still get
    through: a guard that latched on permanently would silence the console
    for the rest of the session and look exactly like a fix.
    """
    import logging

    vl = _restore_console_target
    calls = []

    class _QuietConsole:
        def append_stdout(self, text):
            calls.append(text)

    console = _QuietConsole()     # a strong ref: the target is held weakly
    vl.register_console_target(console)
    vl.apply_verbose_logging(True)
    try:
        for index in range(3):
            logging.getLogger("spacr.trace").debug("marker-%d", index)
            qtbot.wait(5)
    finally:
        vl.apply_verbose_logging(False)

    delivered = "".join(calls)
    for index in range(3):
        assert f"marker-{index}" in delivered, (
            f"line {index} never reached the console; the re-entrancy latch "
            "must reopen after every delivery")


def test_the_real_console_survives_the_function_trace_hook(
        qtbot, qt_theme_applied, _restore_console_target):
    """End-to-end: the real widget, the real profile hook, no recursion.

    Reproduces what ``pytest tests/qt/test_all_module_smoke.py
    tests/qt/test_batch_f_diagnostics.py`` did (SIGSEGV, exit 139) without
    needing the whole shard: a registered ConsolePanel plus verbose
    logging, which switches on ``spacr.logging_util``'s function trace.
    """
    import logging

    from spacr.qt.widgets.console_panel import ConsolePanel, _StdoutBlock

    vl = _restore_console_target
    panel = ConsolePanel()
    qtbot.addWidget(panel)
    vl.register_console_target(panel)

    depth = {"now": 0, "max": 0}
    original = _StdoutBlock.append

    def _counting_append(self, text):
        depth["now"] += 1
        depth["max"] = max(depth["max"], depth["now"])
        try:
            return original(self, text)
        finally:
            depth["now"] -= 1

    _StdoutBlock.append = _counting_append
    vl.apply_verbose_logging(True)
    try:
        for index in range(5):
            logging.getLogger("spacr.trace").debug("trace line %d", index)
            qtbot.wait(5)
    finally:
        vl.apply_verbose_logging(False)
        _StdoutBlock.append = original

    assert depth["max"] <= 1, (
        f"_StdoutBlock.append re-entered {depth['max']} deep; a nested "
        "setPlainText destroys the QTextDocument's frames twice")


def test_the_function_trace_never_fires_on_qt_event_delivery():
    """The other end of the live-lock, and the more important one.

    ``spacr.qt.button_roles.eventFilter`` runs once per delivered Qt event.
    Tracing it emits a log record per event, every record is written into
    the console panel, and writing a widget posts a repaint — which is
    another event. The loop runs through the event queue rather than the
    call stack, so no re-entrancy latch can see it: the GUI thread simply
    never drains its own queue. Captured as a 25-minute stall at 100% CPU
    with the stack parked in ``eventFilter -> _trace_profile ->
    ConsolePanel.append_stdout``.
    """
    import logging as _logging

    from spacr import logging_util

    records = []

    class _Sink(_logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    from spacr.qt import button_roles

    # The real filter, on the real hot path: it is installed on the
    # QApplication, so Qt calls it for every event on every push button.
    filt = button_roles._SemanticButtonFilter()
    sink = _Sink()
    trace = _logging.getLogger("spacr.trace")
    trace.addHandler(sink)
    trace.setLevel(_logging.DEBUG)
    logging_util.enable_function_trace()
    try:
        filt.eventFilter(QObject(), None)
        assert button_roles.action_role("Run") is not None
    finally:
        logging_util.disable_function_trace()
        trace.removeHandler(sink)

    joined = " ".join(records)
    assert "eventFilter" not in joined, (
        "the function trace fired on Qt event delivery; every delivered "
        "event then writes the console, and the console's repaint is "
        "another event")
    # …but it must still trace ordinary application code, or the whole
    # feature has been switched off rather than made safe.
    assert "action_role" in joined, (
        "the exclusion swallowed ordinary spaCR calls too")
    assert "paintEvent" in logging_util._TRACE_SKIP_NAMES


# ---------------------------------------------------------------------------
# 7. Appending to the console must cost what the new text costs
# ---------------------------------------------------------------------------

def test_appending_a_console_line_does_not_rebuild_the_document(
        qtbot, qt_theme_applied):
    """The structural half of the live-lock fix.

    ``append`` used to do ``setPlainText("".join(buf))`` plus a
    document-wide ``mergeBlockFormat`` on every line, which is O(document)
    per line and therefore O(n²) over a run. Asserting the calls rather
    than the clock is what makes this test deterministic; the timing test
    below says why it matters.
    """
    from PySide6.QtWidgets import QPlainTextEdit

    from spacr.qt.widgets.console_panel import _StdoutBlock

    block = _StdoutBlock()
    qtbot.addWidget(block)
    block.append("first line\n")

    calls = []
    original_set = QPlainTextEdit.setPlainText
    original_all = _StdoutBlock._apply_line_spacing

    def _spy_set(self, text):
        calls.append("setPlainText")
        return original_set(self, text)

    def _spy_all(self):
        calls.append("_apply_line_spacing")
        return original_all(self)

    QPlainTextEdit.setPlainText = _spy_set
    _StdoutBlock._apply_line_spacing = _spy_all
    try:
        block.append("second line\n")
    finally:
        QPlainTextEdit.setPlainText = original_set
        _StdoutBlock._apply_line_spacing = original_all

    assert calls == [], (
        f"append() reached {calls}; both rewrite the whole document, which "
        "makes a run's own console output quadratic in its length")
    assert "first line" in block.text()
    assert "second line" in block.text()


def test_the_console_stays_linear_and_keeps_its_cap(qtbot, qt_theme_applied):
    """The clock half: many lines must not take superlinear time.

    Measured on this tree before the fix: 0.56 ms per line for the first
    500 and 6.64 ms by line 3000 — 9.6 s for 3000 lines, still climbing.
    After: flat at ~0.1 ms, 1.3 s for 12000. The budget below is ~20x the
    fixed cost and a small fraction of the old one, so it distinguishes the
    two without being a benchmark.
    """
    from spacr.qt.widgets.console_panel import _StdoutBlock

    block = _StdoutBlock()
    qtbot.addWidget(block)
    block.resize(900, 400)
    line = "[12:00:00] spacr.trace DEBUG  -> spacr.qt.widgets.console.x\n"

    start = time.monotonic()
    for index in range(6000):
        block.append(f"{index:05d} {line}")
    elapsed = time.monotonic() - start

    assert elapsed < 15.0, (
        f"6000 console lines took {elapsed:.1f}s; append() is rebuilding "
        "the document rather than adding to it")
    text = block.text()
    assert len(text) <= _StdoutBlock.MAX_CHARS + len(line) + 8, (
        "the head-trimming cap stopped holding")
    assert "05999" in text, "the newest line was trimmed instead of the oldest"
    assert "00000" not in text, "the oldest line survived past the cap"


# ---------------------------------------------------------------------------
# 8. Screens must retire their own jobs, and drain them on close
# ---------------------------------------------------------------------------

def test_prune_job_pairs_treats_a_reaped_wrapper_as_retired():
    """The tolerance the screens' retirement depends on, asserted directly."""
    class _Reaped:
        def isRunning(self):
            raise RuntimeError("Internal C++ object already deleted.")

    class _Running:
        def isRunning(self):
            return True

    class _Stopped:
        def isRunning(self):
            return False

    from spacr.qt.bridge import prune_job_pairs

    running = (_Running(), object())
    finished = (_Running(), object())
    pairs = [(_Reaped(), object()), running, (_Stopped(), object()), finished]

    # A wrapper Qt already deleted is proof of retirement, a stopped thread
    # is retired, and the sender is retired even while it says it is running
    # (Qt emits `finished` from inside the thread's own teardown).
    assert prune_job_pairs(pairs, finished[0]) == [running]
    # …and with no sender the same sweep still drops the dead ones.
    assert prune_job_pairs(pairs) == [running, finished]


def test_run_history_retires_its_job_when_the_thread_stops(
        qtbot, qt_theme_applied, monkeypatch):
    """A finished refresh must not stay in the screen's ownership list.

    Retiring by ``isRunning()`` alone leaked every pair: this slot is queued
    onto the GUI thread and the pump that delivers it has already flushed
    ``thread.finished -> deleteLater``, so ``isRunning()`` raises
    ``RuntimeError`` inside the slot and the list is never reassigned.
    """
    from spacr.qt.screens.run_history import RunHistoryScreen

    monkeypatch.setattr(
        "spacr.qt.screens.run_history.search_runs", lambda: [])
    screen = RunHistoryScreen(threaded=True)
    qtbot.addWidget(screen)
    screen.refresh()
    assert screen._jobs, "refresh() did not retain its worker pair"
    thread = screen._jobs[0][0]
    _join(qtbot, thread)
    qtbot.waitUntil(lambda: not screen._jobs, timeout=5000)


def test_run_history_close_drains_its_worker(
        qtbot, qt_theme_applied, monkeypatch):
    """A screen that goes away must not leave an ownerless job behind."""
    from spacr.qt.screens.run_history import RunHistoryScreen

    release = threading.Event()
    entered = threading.Event()

    def _slow_search():
        entered.set()
        release.set()
        return []

    monkeypatch.setattr(
        "spacr.qt.screens.run_history.search_runs", _slow_search)
    screen = RunHistoryScreen(threaded=True)
    qtbot.addWidget(screen)
    screen.refresh()
    assert entered.wait(5)
    screen.close()
    assert screen._jobs == [], (
        "closeEvent must drain the history worker; an ownerless job stays "
        "in the run registry and is read as 'analysis still running'")


def test_make_masks_close_drains_the_background_loader(
        qtbot, qt_theme_applied, monkeypatch):
    """``_MaskLoadWorker`` is parented to the screen — destroying the screen
    while it decodes deletes a running QThread, which Qt answers with
    ``qFatal``, not an exception."""
    from spacr.qt.screens import make_masks as mm

    release = threading.Event()
    entered = threading.Event()

    def _slow_load(_folder, _filename):
        entered.set()
        release.wait(20)
        import numpy as np
        return np.zeros((4, 4), "uint16"), np.zeros((4, 4), "uint8")

    monkeypatch.setattr(mm.engine, "load_image_and_mask", _slow_load)
    screen = mm.MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._start_background_load("/nowhere", "a.tif", 1)
    assert entered.wait(5)
    worker = screen._load_worker
    assert worker is not None and worker.isRunning()

    release.set()
    screen.close()
    assert screen._load_worker is None
    assert not worker.isRunning(), (
        "closeEvent must join the mask loader before Qt destroys the screen")


def test_queue_screen_close_stops_the_runner(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """``_QueueRunner`` runs whole pipelines and is parented to the screen.

    It never went through ``make_thread``, so the process-wide run registry
    — and therefore ``MainWindow.closeEvent``'s drain — has never been able
    to see it. Destroying the screen with it running deletes a live QThread,
    which Qt answers with ``qFatal``.
    """
    import spacr.qt.bridge as bridge_mod
    from spacr.qt.plate_queue import PlateQueue, QueueItem, Status
    from spacr.qt.screens import queue as queue_mod

    entered = threading.Event()

    def _endless(_settings):
        entered.set()
        while True:
            cancellation_checkpoint()
            time.sleep(0.002)

    monkeypatch.setattr(
        bridge_mod, "resolve_pipeline_entry", lambda _key: _endless)

    plate_queue = PlateQueue(path=tmp_path / "queue.json")
    plate_queue.add(QueueItem.build("mask", {"src": str(tmp_path)}, label="x"))
    screen = queue_mod.QueueScreen(plate_queue)
    qtbot.addWidget(screen)
    screen.start_runner()
    runner = screen._runner
    assert runner is not None
    assert entered.wait(5), "the runner never reached the pipeline entry"

    screen.close()

    assert screen._runner is None
    assert not runner.isRunning(), (
        "closeEvent must stop the queue runner; destroying the screen with "
        "it running deletes a live QThread")
    # Cancelled at a safe boundary, so the item is left runnable rather than
    # marked failed — Stop is not an error.
    assert plate_queue.items()[0].status == Status.QUEUED
