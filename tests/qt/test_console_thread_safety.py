"""A log record from a worker thread must not build a QWidget there.

Python's logging module calls handlers **inline, on whatever thread
logged the record**. :class:`spacr.qt.widgets.console_panel.ConsolePanel`
answers ``append_stdout`` by constructing widgets — a ``_TopicBar`` and a
``_StdoutBlock``. Qt forbids constructing a QWidget anywhere but the GUI
thread, so the two facts together were a crash waiting for a caller.

The caller was
:meth:`spacr.qt.hf_download._HFDownloadWorker.run`, whose failure path
does ``LOG.warning(..., exc_info=True)`` from inside the download thread::

    worker thread -> logging.warning
                  -> verbose_logger._ConsoleForwarder.emit
                  -> ConsolePanel.append_stdout
                  -> begin_topic -> _TopicBar(...)      # QWidget, wrong thread

Measured before the fix, driving the real ``download_toxo_mito_demo``
against a ``_list_files`` that raises: Qt printed ``QObject::setParent:
Cannot set parent, new parent is in a different thread`` twice and both
console entries were constructed on the download thread. It killed the
test process at
``test_hf_download.py::test_demo_download_reports_a_network_failure_through_the_callback``
when that test ran after the rest of the suite, and passed in isolation.

Two independent layers are asserted here, because either one alone would
leave a hole:

1. :class:`spacr.qt.verbose_logger._ConsoleRelay` hops the formatted line
   onto the GUI thread through a queued signal, so nothing downstream of
   the logging handler runs on the worker thread.
2. ``ConsolePanel.append_stdout`` / ``append_error`` re-post themselves
   when they are entered off-thread, so *any* caller is safe — not just
   the logging one.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import pytest

from PySide6.QtCore import QCoreApplication, QThread
from PySide6.QtWidgets import QWidget
from shiboken6 import isValid       # ships with PySide6

from spacr.qt import hf_download as hf
from spacr.qt import verbose_logger as vl
from spacr.qt.widgets import console_panel as cp


@pytest.fixture(autouse=True)
def _restore_console_target():
    """``_console_ref`` is process-wide; never leak a panel between tests."""
    before = vl._console_ref
    yield
    vl._console_ref = before


@pytest.fixture
def widget_threads(monkeypatch):
    """Record the thread every console entry widget is constructed on."""
    seen: list = []
    for cls in (cp._TopicBar, cp._StdoutBlock):
        original = cls.__init__

        def spy(self, *args, _orig=original, _name=cls.__name__, **kwargs):
            seen.append((_name, threading.get_ident()))
            return _orig(self, *args, **kwargs)
        monkeypatch.setattr(cls, "__init__", spy)
    return seen


def _off_gui(seen, gui_thread):
    return [entry for entry in seen if entry[1] != gui_thread]


# ===========================================================================
# 1. The reported crash, end to end
# ===========================================================================

def test_a_failed_download_logs_from_its_worker_thread(qtbot, tmp_path,
                                                       monkeypatch,
                                                       widget_threads):
    """The whole path, driven by the shipped entry point.

    ``download_toxo_mito_demo`` -> worker thread -> ``LOG.warning`` ->
    console. The console entries must exist (the user has to see the
    failure) and must have been built on the GUI thread.
    """
    gui_thread = threading.get_ident()
    panel = cp.ConsolePanel("mask")
    qtbot.addWidget(panel)
    vl.register_console_target(panel)

    def offline(repo, sub):
        raise ConnectionError("Max retries exceeded with url: /api/datasets")
    monkeypatch.setattr(hf, "_list_files", offline)

    parent = QWidget()
    qtbot.addWidget(parent)
    outcome: list = []
    hf.download_toxo_mito_demo(parent, tmp_path,
                               lambda r, e: outcome.append((r, e)))
    qtbot.waitUntil(lambda: bool(outcome), timeout=10000)
    # The relay is queued — let the GUI thread drain it.
    qtbot.waitUntil(lambda: bool(widget_threads), timeout=10000)

    assert outcome[0][0] is None
    # The callback gets the explained message, not the raw exception text:
    # `explain_download_failure` turns a connection error into a sentence
    # naming huggingface.co and pointing at the offline synthetic demos. The
    # raw "Max retries exceeded" still reaches the LOG.warning below, which is
    # what this test is actually about.
    assert "Could not reach huggingface.co" in outcome[0][1]
    assert not _off_gui(widget_threads, gui_thread), (
        "console entries were constructed off the GUI thread: "
        f"{_off_gui(widget_threads, gui_thread)}")
    text = " ".join(b.text() for b in panel.findChildren(cp._StdoutBlock))
    assert "hf download failed" in text, \
        "the user never saw the download failure"


def test_the_download_worker_is_never_scheduled_for_deletion(qtbot, tmp_path,
                                                             monkeypatch):
    """``thread.finished.connect(worker.deleteLater)`` is the measured bug.

    :func:`spacr.qt.bridge.make_thread` documents it: the worker's
    affinity is the worker thread, so the deferred delete is flushed by
    a loop that is shutting down while the GUI thread still holds the
    only Python reference. Two owners, one object. Chaining off
    ``thread.finished`` instead of ``worker.finished`` was measured at 2
    crashes in 20 runs and is *not* a fix.

    So: once the flow is over and the thread has exited, the worker's
    C++ half is still alive. Python frees it, on the thread that holds
    it.
    """
    monkeypatch.setattr(hf, "_list_files", lambda repo, sub: [])
    parent = QWidget()
    qtbot.addWidget(parent)
    outcome: list = []
    hf.download_toxo_mito_demo(parent, tmp_path,
                               lambda r, e: outcome.append((r, e)))
    worker = parent._hf_download_worker      # keep it alive ourselves
    thread = parent._hf_download_thread

    qtbot.waitUntil(lambda: bool(outcome), timeout=10000)
    qtbot.waitUntil(lambda: not thread.isRunning(), timeout=10000)
    qtbot.wait(50)                           # let any deferred delete flush

    assert isValid(worker), (
        "the worker's C++ half was deleted while Python still held it — "
        "someone re-added worker.deleteLater")


# ===========================================================================
# 2. The relay
# ===========================================================================

def test_the_relay_lives_on_the_gui_thread_even_if_built_elsewhere(qapp,
                                                                   monkeypatch):
    """Affinity must not depend on which thread logged first.

    A worker thread can be the first to touch the relay (that is exactly
    the crashing scenario), and an object created there would deliver
    inline on that thread — the bug, reintroduced through the fix.
    """
    monkeypatch.setattr(vl, "_relay", None)
    built: dict = {}

    class Builder(QThread):
        def run(self):
            built["relay"] = vl._ensure_relay()

    t = Builder()
    t.start()
    assert t.wait(10000)

    relay = built["relay"]
    assert relay is vl._ensure_relay(), "the relay is not a singleton"
    assert relay.thread() is qapp.thread()
    assert relay.thread() is not t


def test_a_record_logged_off_thread_reaches_the_console_on_the_gui_thread(
        qtbot, widget_threads):
    """The logging handler itself, without the download machinery."""
    gui_thread = threading.get_ident()
    panel = cp.ConsolePanel()
    qtbot.addWidget(panel)
    vl.register_console_target(panel)
    vl._ensure_handler()

    class Logger(QThread):
        def run(self):
            logging.getLogger("spacr.qt.hf_download").warning(
                "worker-thread breadcrumb")

    t = Logger()
    t.start()
    assert t.wait(10000)

    qtbot.waitUntil(lambda: bool(widget_threads), timeout=10000)
    assert not _off_gui(widget_threads, gui_thread)
    text = " ".join(b.text() for b in panel.findChildren(cp._StdoutBlock))
    assert "worker-thread breadcrumb" in text


def test_a_record_logged_on_the_gui_thread_is_delivered_synchronously():
    """Direct connection on the GUI thread — no event-loop round trip.

    The console is where a user watches a run; a queued hop for every
    line would reorder output against anything appended directly.
    """
    class FakeConsole:
        def __init__(self):
            self.lines: list = []

        def append_stdout(self, text):
            self.lines.append(text)

    fake = FakeConsole()
    vl.register_console_target(fake)
    vl._ensure_handler()
    logging.getLogger("spacr.qt.hf_download").warning("same-thread line")
    assert any("same-thread line" in line for line in fake.lines)


class _Recorder:
    """A console stand-in that only remembers what it was handed.

    Deliberately holds nothing but the sink list: the *instance* stays
    collectable so a test can drop it and still read what was (not)
    written to it afterwards.
    """

    def __init__(self, sink):
        self._sink = sink

    def append_stdout(self, text):
        self._sink.append(text)


def test_a_collected_console_target_is_dropped_rather_than_resurrected():
    """The weak reference still holds after the relay was introduced.

    Raising from the dead console's ``append_stdout`` would prove
    nothing — both sinks swallow every exception on purpose — so the
    delivery is *recorded* instead. The second half of the test is the
    control: the identical console, identical log call, one strong
    reference, and the recording comes out non-empty.
    """
    delivered: list = []

    vl.register_console_target(_Recorder(delivered))   # no strong reference
    vl._ensure_handler()
    import gc
    gc.collect()
    assert vl._console_ref is not None
    assert vl._console_ref() is None, "the console was kept alive by the relay"

    logging.getLogger("spacr.qt.hf_download").warning("into the void")
    assert delivered == [], f"a collected console was written to: {delivered}"

    # Control — same class, same logger, same sink, one strong reference.
    live = _Recorder(delivered)
    vl.register_console_target(live)
    gc.collect()
    assert vl._console_ref() is live
    logging.getLogger("spacr.qt.hf_download").warning("still alive")
    assert any("still alive" in line for line in delivered), (
        "a live console got nothing either — the measurement is blind")


def test_the_relay_swallows_an_exploding_console(qtbot):
    """A broken console must never take the logging call down with it.

    Asserted through what survives it, because "did not raise" is not
    observable: the exploding console really was reached, a *second*
    record still reaches it (one failure does not latch it off), a
    target with no ``append_stdout`` is skipped rather than
    deregistered, and the relay/handler that carried all that are the
    same objects afterwards and still deliver to a healthy console.
    """
    log = logging.getLogger("spacr.qt.hf_download")
    relay_before = vl._ensure_relay()
    seen: list = []

    class Angry:
        def append_stdout(self, text):
            seen.append(text)
            raise RuntimeError("Internal C++ object already deleted.")

    angry = Angry()
    vl.register_console_target(angry)
    vl._ensure_handler()
    log.warning("first-boom")
    assert any("first-boom" in text for text in seen), (
        "the exploding console was never reached — nothing was swallowed")
    # NB: one record arrives more than once — the same handler is attached to
    # every logger in the `spacr` -> `spacr.qt` -> `spacr.qt.hf_download`
    # chain, so `callHandlers` runs it once per ancestor. Count messages,
    # not deliveries.
    log.warning("second-boom")
    assert any("second-boom" in text for text in seen), (
        f"the first failure latched the console off: {seen}")
    assert vl._console_ref() is angry, (
        "a console that raised once was deregistered")

    # And a target with no append_stdout at all is simply skipped.
    class Mute:
        pass
    mute = Mute()
    vl.register_console_target(mute)
    log.warning("also fine")
    assert vl._console_ref() is mute, (
        "a target without append_stdout was dropped instead of skipped")

    # Control: everything above ran through the live machinery, and it is
    # still the same machinery — the next console gets its line.
    healthy: list = []
    good = _Recorder(healthy)
    vl.register_console_target(good)
    log.warning("after the boom")
    assert vl._ensure_relay() is relay_before, "the relay was rebuilt"
    assert vl._handler in log.handlers, "the handler fell off the logger"
    assert any("after the boom" in text for text in healthy), (
        "the broken console took the whole sink down with it")


# ===========================================================================
# 3. The panel's own guard
# ===========================================================================

@pytest.mark.parametrize("method,probe", [
    ("append_stdout", "stdout from a worker"),
    ("append_error", "Traceback from a worker"),
])
def test_the_panel_reposts_an_off_thread_append(qtbot, widget_threads,
                                                method, probe):
    """Any caller, not just the logging one, is bounced to the GUI thread."""
    gui_thread = threading.get_ident()
    panel = cp.ConsolePanel()
    qtbot.addWidget(panel)

    class Caller(QThread):
        def run(self):
            getattr(panel, method)(probe)

    t = Caller()
    t.start()
    assert t.wait(10000)

    qtbot.waitUntil(lambda: bool(widget_threads), timeout=10000)
    assert not _off_gui(widget_threads, gui_thread)
    text = " ".join(b.text() for b in panel.findChildren(cp._StdoutBlock))
    assert probe in text


def test_the_guard_reports_the_gui_thread_correctly(qtbot):
    """``_on_gui_thread`` is the whole fix — assert it directly."""
    panel = cp.ConsolePanel()
    qtbot.addWidget(panel)
    assert panel._on_gui_thread() is True

    answers: list = []

    class Asker(QThread):
        def run(self):
            answers.append(panel._on_gui_thread())

    t = Asker()
    t.start()
    assert t.wait(10000)
    assert answers == [False]


def test_an_empty_append_is_still_a_no_op_from_a_worker_thread(qtbot,
                                                               widget_threads):
    """The empty-string early return must come before the relay hop."""
    panel = cp.ConsolePanel()
    qtbot.addWidget(panel)

    class Caller(QThread):
        def run(self):
            panel.append_stdout("")
            panel.append_error("")

    t = Caller()
    t.start()
    assert t.wait(10000)
    qtbot.wait(100)
    assert widget_threads == []


def test_hf_download_still_reports_the_failure_text(qapp, tmp_path,
                                                    monkeypatch, caplog):
    """The warning that triggers all of this is worth keeping.

    Two audiences, two texts, and they are deliberately different. The *log*
    keeps the raw exception with its traceback — that is what a bug report is
    read from. The *signal* carries what ``explain_download_failure`` made of
    it, because that string goes straight into a QMessageBox and "no dns" is
    not something a user can act on.
    """
    monkeypatch.setattr(
        hf, "_list_files",
        lambda repo, sub: (_ for _ in ()).throw(ConnectionError("no dns")))
    worker = hf._HFDownloadWorker(Path(tmp_path))
    seen: list = []
    worker.finished.connect(lambda *a: seen.append(a))
    with caplog.at_level(logging.WARNING, logger="spacr.qt.hf_download"):
        worker.run()
    assert len(seen) == 1, seen
    ok, dataset, settings, error = seen[0]
    assert (ok, dataset, settings) == (False, "", "")
    assert "Could not reach huggingface.co" in error
    assert "internet connection" in error
    assert any("hf download failed" in r.message for r in caplog.records)
    assert any("no dns" in str(r.getMessage()) for r in caplog.records), (
        "the raw exception must survive into the log even though the dialog "
        "gets the explained version")
