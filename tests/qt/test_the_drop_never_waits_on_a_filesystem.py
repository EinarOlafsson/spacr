"""A drop classifies its paths off the GUI thread.

THE FREEZE, 2026-09-04. ``_DropzoneFilter._on_drop`` ran inside Qt's delivery
of ``QEvent.Drop`` -- on the GUI thread, by definition, because that is the
thread that owns the widget -- and the first thing it did was touch the disk:

    _DropzoneFilter._on_drop
      -> p.is_file()                         to split the settings CSVs out
      -> handler.can_accept(p)               -> dnd.has_images_in
                                                -> Path.is_dir + iterdir
      -> handler.suggest_alternatives(p)     -> dnd.find_image_folders_nearby
                                                -> two more levels of iterdir
      -> _apply_settings_csv(p, screen)      -> open(path)  (and load_settings,
                                                which reads the whole file)

and EVERY path in that chain came from ``event.mimeData()`` -- it is whatever
the user dragged out of their file manager. One of the maintainer's file
managers was showing a ``/nas_mnt`` share behind an ``autofs`` mount, and a
single ``os.path.exists`` on a sleeping one had not returned after TWENTY
SECONDS. Dragging a folder from it froze the whole application, with no
traceback, because a stalled event loop is not a crash. See
:mod:`spacr.qt.path_probe` for the rest of that story; the dropzone is
installed on roughly forty-five screens, so this was every one of them.

The fix splits the drop in two: ``_classify_drop`` asks the disk everything
on the screen's drop scanner, and ``_deliver_drop`` does the widget work when
the answers come back. These tests hold the seam. Each blocks the primitive
the way a sleeping mount does and asserts the drop event returns anyway --
and then lets the block go and asserts the user still gets every row, warning
and applied path they got before, a moment later.
"""
from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import (
    QEventLoop, QMimeData, QPoint, QPointF, QTimer, QUrl, Qt,
)
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import QApplication, QWidget

from spacr.qt import dnd as dnd_mod
from spacr.qt.dnd import DropHandler, install_dropzone

#: Longer than any human would call responsive, shorter than the twenty
#: seconds actually measured. A test that waited the real duration would be
#: a test nobody runs.
SLOW_S = 8.0

#: What "did not block" means. The drop event has to come back in the time it
#: takes to build a closure, not in the time it takes to wake a mount.
FAST_S = 1.0


class _Gate:
    """A stat that hangs, and a way to stop hanging.

    ``wait`` is what the sleeping mount does: the caller is parked for up to
    :data:`SLOW_S`. ``open`` is what the test does once it has proved the GUI
    thread got past it -- without that, every one of these tests would cost
    eight seconds to assert something that takes milliseconds, and the second
    half (the user still gets their answer) could not be asserted at all.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self.threads: list = []

    def wait(self) -> None:
        self.threads.append(threading.current_thread())
        self._event.wait(SLOW_S)

    def open(self) -> None:
        self._event.set()


class _Screen(QWidget):
    """The little a drop asks of the screen it lands on."""

    def __init__(self) -> None:
        super().__init__()
        self.applied: list = []
        self.reported: list = []

    def apply_settings_dict(self, values):
        self.applied.append(dict(values))
        return len(values)

    def _set_status(self, text):
        self.reported.append(text)


class _GatedHandler(DropHandler):
    """A policy whose filesystem questions hang, as a sleeping share's do."""

    def __init__(self, gate: _Gate, accept: bool = True) -> None:
        self._gate = gate
        self._accept = accept
        self.applied: list = []

    def can_accept(self, path):
        self._gate.wait()
        return self._accept

    def suggest_alternatives(self, path):
        return []

    def error_message(self, path):
        return f"cannot use {path.name}"

    def apply(self, path, screen):
        self.applied.append(path)


class _MessageBox:
    """Records instead of blocking; the modal guard makes the real one raise."""

    def __init__(self) -> None:
        self.calls: list = []

    def information(self, parent, title, text):
        self.calls.append(("information", title, text))

    def warning(self, parent, title, text):
        self.calls.append(("warning", title, text))


@pytest.fixture
def msgbox(monkeypatch):
    box = _MessageBox()
    monkeypatch.setattr(dnd_mod, "QMessageBox", box)
    return box


@pytest.fixture
def dropzone(qtbot, qt_theme_applied):
    """Factory: a shown widget with a dropzone, torn down with its scanner.

    The scanner has to be shut down explicitly. Qt ABORTS THE PROCESS when a
    running QThread is destroyed, and these tests deliberately leave a worker
    parked inside a gate.
    """
    made = []

    def _make(handler):
        screen = _Screen()
        qtbot.addWidget(screen)
        screen.resize(120, 120)
        screen.show()
        install_dropzone(screen, handler, screen)
        made.append(screen)
        return screen

    yield _make
    for screen in made:
        scanner = getattr(screen, "_dnd_scanner", None)
        if scanner is not None:
            scanner.shutdown()
        screen.hide()


def _drop(widget, paths):
    """Replay enter -> move -> drop and return the drop event, unsettled.

    Deliberately NOT the ``_drop`` in ``test_dnd_dropzone.py``: that one
    waits for the classification to come back, which is the very thing these
    tests are timing.
    """
    urls = [QUrl.fromLocalFile(str(p)) for p in paths]

    def mime():
        m = QMimeData()
        m.setUrls(urls)
        return m

    # Each QMimeData is bound to a NAME and kept alive until the function
    # returns. A Qt event does not own the mime data handed to it, so a
    # temporary is freed the moment the expression ends and the event filter
    # then reads a dangling pointer -- which segfaults the interpreter rather
    # than raising, and only once something actually calls mimeData().
    enter_mime, move_mime, drop_mime = mime(), mime(), mime()
    QApplication.sendEvent(widget, QDragEnterEvent(
        QPoint(5, 5), Qt.CopyAction, enter_mime, Qt.LeftButton,
        Qt.NoModifier))
    QApplication.sendEvent(widget, QDragMoveEvent(
        QPoint(5, 5), Qt.CopyAction, move_mime, Qt.LeftButton, Qt.NoModifier))
    event = QDropEvent(QPointF(5, 5), Qt.CopyAction, drop_mime,
                       Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, event)
    return event


def _settings_csv(path):
    path.write_text("Key,Value\nsrc,/plate01\nnr,3\n")
    return path


# ---------------------------------------------------------------------------
# The property the freeze violated
# ---------------------------------------------------------------------------

def test_a_dropped_folder_does_not_wait_for_can_accept(dropzone, tmp_path,
                                                       qtbot, msgbox):
    """The freeze itself: ``can_accept`` walks the folder, on a mount."""
    gate = _Gate()
    handler = _GatedHandler(gate)
    screen = dropzone(handler)
    folder = tmp_path / "plate01"
    folder.mkdir()

    started = time.monotonic()
    event = _drop(screen, [folder])
    elapsed = time.monotonic() - started

    assert elapsed < FAST_S, (
        f"the drop event took {elapsed:.1f}s -- the dropzone is classifying "
        "on the GUI thread again, which is the freeze")
    # ...and the OS is told the drop landed regardless, so the drag animation
    # does not snap back while the scan is still out.
    assert event.isAccepted() is True

    # The second half of the promise: the user loses nothing, it just arrives
    # a moment later.
    gate.open()
    qtbot.waitUntil(lambda: handler.applied == [folder], timeout=10000)
    assert msgbox.calls == []


def test_a_dropped_settings_csv_is_not_read_on_the_gui_thread(
        dropzone, tmp_path, qtbot, monkeypatch, msgbox):
    """``_csv_header`` opens the dropped file; ``load_settings`` reads it all."""
    gate = _Gate()
    real_header = dnd_mod._csv_header

    def gated_header(path):
        gate.wait()
        return real_header(path)

    monkeypatch.setattr(dnd_mod, "_csv_header", gated_header)

    screen = dropzone(_GatedHandler(_Gate()))
    csv = _settings_csv(tmp_path / "settings.csv")

    started = time.monotonic()
    event = _drop(screen, [csv])
    elapsed = time.monotonic() - started

    assert elapsed < FAST_S, (
        f"the drop event took {elapsed:.1f}s -- the settings CSV is being "
        "read on the GUI thread")
    assert event.isAccepted() is True

    gate.open()
    qtbot.waitUntil(lambda: bool(screen.applied), timeout=10000)
    assert screen.applied[0]["src"] == "/plate01"


def test_the_disk_questions_run_on_a_worker_and_not_the_gui_thread(
        dropzone, tmp_path, qtbot, msgbox):
    """Timing says "fast"; this says WHERE, so a fast local disk cannot lie.

    A future change that put the classification back on the GUI thread would
    still pass a timing assertion on a laptop SSD. It would fail this.
    """
    gate = _Gate()
    handler = _GatedHandler(gate)
    screen = dropzone(handler)
    folder = tmp_path / "plate01"
    folder.mkdir()

    _drop(screen, [folder])
    gate.open()
    qtbot.waitUntil(lambda: handler.applied == [folder], timeout=10000)

    assert gate.threads, "can_accept was never called"
    assert threading.main_thread() not in gate.threads, (
        "can_accept ran on the GUI thread -- that is the stat that froze the "
        "application for twenty seconds")


def test_a_rejected_drop_still_reports_after_the_scan(dropzone, tmp_path,
                                                      qtbot, msgbox):
    """Rule two: every warning the user got before, they still get."""
    gate = _Gate()
    handler = _GatedHandler(gate, accept=False)
    screen = dropzone(handler)
    folder = tmp_path / "wrong"
    folder.mkdir()

    started = time.monotonic()
    _drop(screen, [folder])
    assert time.monotonic() - started < FAST_S

    gate.open()
    qtbot.waitUntil(lambda: bool(msgbox.calls), timeout=10000)
    kind, title, text = msgbox.calls[0]
    assert kind == "information"
    assert title == "Nothing to drop into"
    assert "cannot use wrong" in text
    assert handler.applied == []
    # The console/status report is part of what the user saw before, too.
    assert any("Drop rejected" in line for line in screen.reported)


def test_a_second_sleeping_path_is_never_even_asked(dropzone, tmp_path,
                                                    qtbot, msgbox):
    """Single-drop modules truncate BEFORE the scan, not after.

    ``others[:1]`` used to happen after the split but before ``can_accept``,
    which was free when the truncation was inline. Off the GUI thread it
    would be tempting to scan everything and discard the extras -- and a
    second sleeping mount is a second twenty-second wait that nothing will
    ever look at.
    """
    gate = _Gate()
    handler = _GatedHandler(gate)
    screen = dropzone(handler)
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()

    _drop(screen, [first, second])
    gate.open()
    qtbot.waitUntil(lambda: handler.applied == [first], timeout=10000)
    assert len(gate.threads) == 1, (
        "the folder the drop was going to discard was stat'd anyway")


# ---------------------------------------------------------------------------
# The edges the move opened, and what closes them
# ---------------------------------------------------------------------------

class _SelectiveGate(DropHandler):
    """Hangs on ONE folder's name, answers instantly for every other."""

    def __init__(self, slow_name: str) -> None:
        self.gate = _Gate()
        self._slow = slow_name
        self.scanned: list = []
        self.applied: list = []

    def can_accept(self, path):
        self.scanned.append(path)
        if path.name == self._slow:
            self.gate.wait()
        return True

    def apply(self, path, screen):
        self.applied.append(path)

    def error_message(self, path):
        return f"cannot use {path.name}"


def test_two_drops_are_delivered_in_the_order_they_were_made(
        dropzone, tmp_path, qtbot, msgbox):
    """Inline, a second drop could not overtake the first. Threaded, it can.

    The scanner runs a thread PER JOB, so two drops finish in the order the
    FILESYSTEM answers. The gesture that produced this: drop a folder from a
    sleeping share, watch nothing happen, drop a local one instead -- and
    twenty seconds later the first lands last and overwrites the source the
    user actually chose. The queue in ``_route_drop`` holds each answer until
    every earlier drop has been delivered, so both are applied and the last
    one dropped is still the last one applied.
    """
    handler = _SelectiveGate("nas")
    screen = dropzone(handler)
    slow, fast = tmp_path / "nas", tmp_path / "local"
    slow.mkdir()
    fast.mkdir()

    _drop(screen, [slow])
    _drop(screen, [fast])

    # The fast drop's scan finishes first -- and is held.
    qtbot.waitUntil(lambda: len(handler.scanned) == 2, timeout=10000)
    qtbot.wait(150)
    assert handler.applied == [], (
        "the second drop was applied while the first was still being "
        "classified -- the sleeping share's folder will overwrite it")

    handler.gate.open()
    qtbot.waitUntil(lambda: len(handler.applied) == 2, timeout=10000)
    assert handler.applied == [slow, fast]


class _AngryHandler(DropHandler):
    """A policy that raises, the way a vanished mount makes one raise."""

    def __init__(self) -> None:
        self.applied: list = []

    def can_accept(self, path):
        raise RuntimeError("the mount went away")

    def apply(self, path, screen):
        self.applied.append(path)


def test_a_policy_that_raises_is_reported_and_not_swallowed(
        dropzone, tmp_path, qtbot, msgbox):
    """A JobRunner calls ``on_done`` only for a job that SUCCEEDED.

    So an exception escaping the classification would take the whole delivery
    with it: no import, no rejection report, no dialog, no status line -- a
    drop that silently did nothing, where the same exception on the GUI
    thread at least printed a traceback. ``_classify_drop`` therefore carries
    the failure back as a rejection.
    """
    handler = _AngryHandler()
    screen = dropzone(handler)
    folder = tmp_path / "gone"
    folder.mkdir()

    _drop(screen, [folder])

    qtbot.waitUntil(lambda: bool(screen.reported), timeout=10000)
    assert any("the mount went away" in line for line in screen.reported)
    assert handler.applied == []


def test_an_unreadable_csv_header_does_not_take_the_drop_with_it(
        dropzone, tmp_path, qtbot, msgbox):
    """``_csv_header`` caught ``OSError`` only; ``csv`` raises its own Error.

    A file named ``.csv`` whose first field runs past
    ``csv.field_size_limit()`` -- which is every binary anyone ever renamed
    -- raised ``_csv.Error``. Inline that went straight out through Qt's
    event delivery; on a worker it fails the JOB, and a failed job never
    reaches ``on_done``, so it would take every OTHER path in the same drop
    down with it silently.
    """
    gate = _Gate()
    gate.open()                      # nothing to stall here; only to observe
    handler = _GatedHandler(gate)
    screen = dropzone(handler)
    bad = tmp_path / "binary.csv"
    bad.write_bytes(b"A" * 200_000 + b"\nsrc,/plate01\n")
    folder = tmp_path / "plate01"
    folder.mkdir()

    _drop(screen, [bad, folder])

    # The folder in the same drop is still classified and still applied...
    qtbot.waitUntil(lambda: handler.applied == [folder], timeout=10000)
    # ...and the unreadable CSV is reported as what it is -- a file with no
    # settings header -- rather than silently discarded.
    qtbot.waitUntil(lambda: bool(screen.reported), timeout=10000)
    assert any("is not a settings CSV" in line for line in screen.reported)
    assert screen.applied == []


class _ThreadRecordingHandler(DropHandler):
    """Records which thread each half of the drop ran on."""

    def __init__(self) -> None:
        self.scanned_on: list = []
        self.applied_on: list = []

    def can_accept(self, path):
        self.scanned_on.append(threading.current_thread())
        return True

    def apply(self, path, screen):
        self.applied_on.append(threading.current_thread())


def test_apply_still_runs_on_the_gui_thread(dropzone, tmp_path, qtbot,
                                            msgbox):
    """The seam has two sides, and the second one is a promise too.

    ``apply`` wires the path into WIDGETS, so it cannot move to a worker: the
    handlers that need to read a folder set the source immediately and submit
    their own scan for the rest. A change that carried apply across with the
    classification would be a Qt-from-a-thread bug, not a speed-up.
    """
    handler = _ThreadRecordingHandler()
    screen = dropzone(handler)
    folder = tmp_path / "plate01"
    folder.mkdir()

    _drop(screen, [folder])
    qtbot.waitUntil(lambda: bool(handler.applied_on), timeout=10000)

    assert threading.main_thread() not in handler.scanned_on
    assert handler.applied_on == [threading.main_thread()]


# ---------------------------------------------------------------------------
# The delivery queue must not become a place drops go to die
# ---------------------------------------------------------------------------

def test_a_drop_cancelled_with_its_screen_does_not_wedge_the_next_one(
        dropzone, tmp_path, qtbot, msgbox):
    """A classification that never comes back must not silence the screen.

    The queue that keeps concurrent drops in order holds each answer until
    every earlier drop has been delivered -- and a slot is filled in by the
    classification's completion handler, which a ``JobRunner`` runs ONLY for
    a job that succeeded and was not cancelled. Closing a screen cancels
    them: ``_DropScanner`` shuts its runner down on the Close event. A spaCR
    screen is CACHED rather than destroyed, so the user comes straight back
    to it -- with an unanswered slot at the head of its queue and every later
    drop parked behind that slot for the life of the window. Drag a folder
    on, nothing happens, no message, ever.
    """
    handler = _SelectiveGate("nas")
    screen = dropzone(handler)
    slow, later = tmp_path / "nas", tmp_path / "local"
    slow.mkdir()
    later.mkdir()

    _drop(screen, [slow])
    qtbot.waitUntil(lambda: handler.scanned == [slow], timeout=10000)

    # Leaving the screen cancels the scan that is still parked in the mount.
    # The gate opens from a timer so the worker can retire while the close is
    # waiting for it, rather than after a three-second drain.
    threading.Timer(0.2, handler.gate.open).start()
    screen.close()
    qtbot.wait(50)
    assert handler.applied == [], (
        "the cancelled classification was delivered after all -- this test "
        "is no longer exercising the wedge it was written for")

    # ...and the user comes back to the screen and drops something else.
    screen.show()
    _drop(screen, [later])
    qtbot.waitUntil(lambda: handler.applied == [later], timeout=10000)


def test_a_drop_written_off_as_abandoned_is_still_delivered_if_it_lands(
        qtbot):
    """Letting go of a slot must not be a second way to lose a drop.

    ``_forget_abandoned`` writes off a slot whose scan can no longer land.
    If that judgement is ever wrong -- the answer arrives anyway -- the drop
    has lost its place in line, and losing a place in line is not the same as
    being cancelled: it is delivered where it stands. Out of order beats not
    at all.
    """
    screen = QWidget()
    qtbot.addWidget(screen)
    delivered = []
    slot = dnd_mod._queue_drop(screen, delivered.append)
    assert slot is not None

    queue = dnd_mod._pending_drops.get(screen)
    dnd_mod._forget_abandoned(screen, queue)
    assert queue == [], "the unanswerable slot was kept"

    dnd_mod._answer_drop(screen, slot, ["late"])
    assert delivered == [["late"]]


def test_a_classification_that_never_reports_back_is_not_swallowed(
        dropzone, tmp_path, qtbot, monkeypatch, msgbox):
    """``_scan_then`` can return without ever calling back.

    It answers False both for a scan it ran inline and for one that RAISED
    there, and in the second case ``on_done`` is never reached.
    ``_classify_drop`` is written never to raise, so this is the guard for
    the day something beneath it does: the drop is REPORTED rather than
    silently dropped, and the screen still takes the next one.
    """
    from spacr.qt import dnd_handlers as dh

    handler = _SelectiveGate("nothing-is-slow-here")
    screen = dropzone(handler)
    folder = tmp_path / "plate01"
    folder.mkdir()

    monkeypatch.setattr(dh, "_scan_then", lambda *a, **k: False)
    _drop(screen, [folder])

    assert any("could not be classified" in line for line in screen.reported)
    assert handler.applied == []

    # The queue is not wedged behind it: the next drop still lands.
    monkeypatch.undo()
    other = tmp_path / "plate02"
    other.mkdir()
    _drop(screen, [other])
    qtbot.waitUntil(lambda: handler.applied == [other], timeout=10000)


# ---------------------------------------------------------------------------
# The other door into the same wrong-source bug: a nested event loop
# ---------------------------------------------------------------------------
#
# The queue above holds a later drop until every earlier one has been
# delivered. What it did not watch is that a DELIVERY OPENS MODAL DIALOGS --
# `QMessageBox.information` for a rejected drop, `suggest_alternatives_dialog`
# for a near-miss, `QMessageBox.warning` for a settings CSV that would not
# load -- and every one of those runs a NESTED Qt event loop. Qt goes on
# dispatching queued signals inside it, including the `_on_settled` of a later
# drop's scan, so `_answer_drop` re-entered `_drain` and delivered the later
# drop in the MIDDLE of the earlier one. The newer folder was applied first
# and the older one's `handler.apply` overwrote it on the way out: the wrong
# source wins, which is the exact outcome the queue exists to prevent.


@pytest.fixture
def scan_in_flight(monkeypatch):
    """Report a scan as still out, which is what a real second drop meets.

    `_queue_drop` calls `_forget_abandoned` first, and that writes off every
    unanswered slot the moment NOTHING is in flight -- correctly, because
    with no scan running no answer can still be coming. A slot-level test
    hand-builds its queue and so has no scanner at all, and without this the
    earlier slot is written off before the later one is even added: the
    ordering under test would never apply, and the test would pass on the
    unfixed code for the wrong reason.
    """
    monkeypatch.setattr(dnd_mod, "_scan_in_flight", lambda screen: True)


def test_a_nested_delivery_does_not_let_a_later_drop_overtake_an_earlier_one(
        qtbot, scan_in_flight):
    """A delivery that spins the event loop keeps its place at the head.

    The slot-level statement of it, with no threads and no timing: the first
    drop's delivery answers the second drop's slot from inside itself, which
    is what a modal dialog's nested loop does. The second delivery must wait
    for the first to RETURN, not merely to start.
    """
    screen = QWidget()
    qtbot.addWidget(screen)
    order = []
    later = {}

    def deliver_first(report):
        order.append("first-in")
        # The modal dialog: the later drop's scan lands inside it.
        dnd_mod._answer_drop(screen, later["slot"], ["second"])
        order.append("first-out")

    first = dnd_mod._queue_drop(screen, deliver_first)
    later["slot"] = dnd_mod._queue_drop(
        screen, lambda report: order.append("second"))
    assert first is not None and later["slot"] is not None

    dnd_mod._answer_drop(screen, first, ["first"])

    assert order == ["first-in", "first-out", "second"], (
        "the later drop was delivered inside the earlier one's delivery -- "
        "the earlier drop's apply now runs last and overwrites it")
    assert dnd_mod._draining == [], "the drain guard was not released"


def test_a_delivery_that_raises_still_releases_the_queue(qtbot,
                                                         scan_in_flight):
    """The guard is released even when a delivery blows up inside it.

    `_run_delivery` swallows what a delivery raises so one bad drop costs
    only itself; the re-entrancy guard has to be given back on that path too,
    or the screen's queue is wedged for the life of the window.
    """
    screen = QWidget()
    qtbot.addWidget(screen)
    order = []
    later = {}

    def deliver_first(report):
        order.append("first")
        dnd_mod._answer_drop(screen, later["slot"], ["second"])
        raise RuntimeError("the dialog blew up")

    first = dnd_mod._queue_drop(screen, deliver_first)
    later["slot"] = dnd_mod._queue_drop(
        screen, lambda report: order.append("second"))

    dnd_mod._answer_drop(screen, first, ["first"])

    assert order == ["first", "second"]
    assert dnd_mod._draining == []
    assert dnd_mod._pending_drops.get(screen) == []


class _ModalOnApply(DropHandler):
    """Applies the first drop inside a nested event loop, as a dialog does.

    ``can_accept`` hangs for every path but the first, so the first drop is
    delivered while the second is still being classified -- and the second's
    scan then finishes DURING the first's apply, which is the ordering the
    real bug needed.
    """

    def __init__(self, first_name: str) -> None:
        self.gate = _Gate()
        self._first = first_name
        self.scanned: list = []
        self.events: list = []

    def can_accept(self, path):
        if path.name != self._first:
            self.gate.wait()
        self.scanned.append(path.name)
        return True

    def error_message(self, path):
        return f"cannot use {path.name}"

    def apply(self, path, screen):
        self.events.append(("in", path.name))
        if path.name == self._first:
            # Let the later scan go, then spin the loop the way an open
            # QMessageBox does -- and keep spinning well past the moment it
            # finishes, so its queued `_on_settled` is certainly dispatched
            # in HERE rather than after we return.
            self.gate.open()
            loop = QEventLoop()
            deadline = time.monotonic() + SLOW_S
            turns = {"after": 0}

            def tick():
                if len(self.scanned) == 2:
                    turns["after"] += 1
                if turns["after"] > 10 or time.monotonic() > deadline:
                    loop.quit()

            timer = QTimer()
            timer.timeout.connect(tick)
            timer.start(10)
            loop.exec()
            timer.stop()
        self.events.append(("out", path.name))


def test_a_scan_landing_inside_a_modal_dialog_waits_its_turn(
        dropzone, tmp_path, qtbot, msgbox):
    """End to end, through the real scanner and the real drop events.

    The first drop's apply holds the GUI thread in a nested event loop while
    the second drop's scan finishes and posts its result. The second drop
    must still be applied AFTER the first has finished, because in the real
    application the tail of that first apply is what sets the source.
    """
    handler = _ModalOnApply("first")
    screen = dropzone(handler)
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()

    _drop(screen, [first])
    _drop(screen, [second])

    qtbot.waitUntil(lambda: len(handler.events) == 4, timeout=15000)
    assert handler.events == [
        ("in", "first"), ("out", "first"),
        ("in", "second"), ("out", "second"),
    ], "the second drop was applied inside the first one's dialog"
    assert dnd_mod._draining == []
