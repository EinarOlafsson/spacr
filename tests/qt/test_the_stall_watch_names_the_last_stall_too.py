"""A stall watch that loses the last stall loses the only one that matters.

`spacr/qt/stall_watch.py` samples the GUI thread while it is stuck and then
counts the innermost frames, because ONE SNAPSHOT NAMES WHERE THE THREAD WAS,
NOT WHERE THE TIME WENT -- and those differ whenever the stall is a loop
rather than a single blocking call. The distribution is the useful half.

IT WAS WRITTEN OUT ONLY WHEN THE NEXT STALL BEGAN. So a process that wedges
once and is then killed -- a run against a sleeping autofs mount, which is
the failure this tool exists for -- left a traceback and NO distribution at
all, because there was never a next stall to trigger the write.
"""
from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from PySide6.QtCore import QTimer                            # noqa: E402
from PySide6.QtWidgets import QApplication                   # noqa: E402


def _watch_one_stall(tmp_path, monkeypatch, seconds=3.0, echo=False):
    """Hold the GUI thread once, then let the loop answer again."""
    from spacr.qt import stall_watch

    log = tmp_path / "stalls.log"
    monkeypatch.setattr(stall_watch, "LOG_PATH", log)
    # A quarter of the real poll, so one short stall still yields several
    # samples without making the test slow.
    monkeypatch.setattr(stall_watch, "POLL_SECONDS", 0.05)

    app = QApplication.instance()
    # echo=False: the watcher writes from a DAEMON THREAD, and writing
    # into a stream pytest is swapping underneath it crashes pytest
    # inside its own capture.py. The file is the record either way.
    watcher = stall_watch.watch_this_application(app, stall_seconds=0.4,
                                                 echo=echo)

    def wedge():
        """Block the GUI thread the way a synchronous call does."""
        time.sleep(seconds * 0.4)

    try:
        QTimer.singleShot(50, wedge)
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.02)
            if log.exists() and "WHERE THAT STALL SPENT" in log.read_text():
                break
        return log.read_text() if log.exists() else ""
    finally:
        # THE WATCHER OUTLIVES THIS TEST UNLESS IT IS STOPPED, and the
        # `qapp` it is installed on is session-scoped, so what leaks here
        # leaks into every test that runs afterwards. Two of these sampling
        # the GUI thread's live frames four times a second is what
        # segfaulted the Slow shard on 2026-09-13, hundreds of tests later,
        # in a figure export that had nothing to do with stall watching.
        if watcher is not None:
            watcher.stop()
            watcher.join(1.0)
        timer = getattr(app, "_spacr_stall_timer", None)
        if timer is not None:
            timer.stop()


@pytest.mark.slow
def test_a_single_stall_is_summarised_without_waiting_for_a_second(
        tmp_path, monkeypatch, qapp):
    """THE LAST STALL OF A SESSION IS THE ONE ANYBODY RUNS THIS FOR.

    A process that freezes and is killed has exactly one stall. If the
    summary waits for the next one it is never written, and what survives is
    a single snapshot -- which names a frame that may merely have been
    executing when the sampler fired.
    """
    text = _watch_one_stall(tmp_path, monkeypatch)

    assert "GUI THREAD STALLED" in text, (
        "the watch did not report the stall at all, so this test is not "
        "measuring what it claims")
    assert "WHERE THAT STALL SPENT ITS TIME" in text, (
        "one stall happened and no distribution was written: the summary is "
        "still waiting for a SECOND stall that a wedged process never has.\n"
        + text[-400:])


@pytest.mark.slow
def test_the_summary_counts_more_than_one_sample(tmp_path, monkeypatch, qapp):
    """A distribution of one sample is a snapshot wearing a percentage.

    The whole reason for sampling through the stall is that a single frame
    is not evidence about where the time went. A summary that reports 1/1
    would satisfy the test above while restoring the defect it was written
    for.
    """
    text = _watch_one_stall(tmp_path, monkeypatch)
    marker = "WHERE THAT STALL SPENT ITS TIME"
    assert marker in text, text[-400:]
    header = text[text.index(marker):text.index(marker) + 120]
    assert "samples at" in header, header
    count = int(header.split("(")[1].split(" samples")[0])
    assert count >= 2, (
        f"the stall was summarised from {count} sample(s), which is a "
        "snapshot with a percent sign on it rather than a distribution")


# ---------------------------------------------------------------------------
# The watch is a diagnostic. Nothing it does may cost more than it reports.
# ---------------------------------------------------------------------------

def test_a_log_it_cannot_write_is_not_an_exception(tmp_path, monkeypatch):
    """`_write` runs on a DAEMON THREAD and may never raise.

    An exception from a daemon thread cannot be caught by anything, is
    printed to a stderr the user is not reading, and would turn a tool for
    diagnosing a freeze into a second source of noise during one. A home
    directory that is read-only, full, or on a share that went away is
    ordinary.
    """
    from spacr.qt import stall_watch

    blocked = tmp_path / "not-a-directory" / "stalls.log"
    blocked.parent.write_text("this is a file, not a folder", encoding="utf-8")
    monkeypatch.setattr(stall_watch, "LOG_PATH", blocked)

    stall_watch._write("a stall nobody will read about\n")

    assert not blocked.exists()


def test_without_qt_the_watch_declines_instead_of_raising(monkeypatch):
    """`SPACR_WATCH_GUI_STALLS=1` is read inside `launch`, but not only there.

    The flag composes with every driver, including headless ones with no
    PySide6 at all, and a diagnostic that refuses to start must not stop
    the thing it was watching.
    """
    import builtins

    from spacr.qt import stall_watch

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "PySide6.QtCore":
            raise ImportError("No module named 'PySide6'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    assert stall_watch.watch_this_application(object()) is None


def test_a_second_watch_stops_the_first_ones_timer(tmp_path, monkeypatch, qapp):
    """TWO WATCHERS SAMPLING THE SAME THREAD IS THE BUG THIS FILE RECORDS.

    Two of them reading the GUI thread's live frames four times a second is
    what segfaulted the Slow shard on 2026-09-13, hundreds of tests later,
    in a figure export that had nothing to do with stall watching. So a
    second install adopts the application and stops what was there -- even
    when the old timer's C++ object has already gone, which is the usual
    way a stale one is found.
    """
    from spacr.qt import stall_watch

    monkeypatch.setattr(stall_watch, "LOG_PATH", tmp_path / "stalls.log")

    class _AlreadyDeleted:
        def stop(self):
            raise RuntimeError("Internal C++ object already deleted")

    app = QApplication.instance()
    previous = getattr(app, "_spacr_stall_timer", None)
    app._spacr_stall_timer = _AlreadyDeleted()
    watcher = None
    try:
        watcher = stall_watch.watch_this_application(app, stall_seconds=60.0,
                                                     echo=False)
        assert watcher is not None
        assert not isinstance(app._spacr_stall_timer, _AlreadyDeleted)
    finally:
        if watcher is not None:
            watcher.stop()
            watcher.join(1.0)
        timer = getattr(app, "_spacr_stall_timer", None)
        if timer is not None and not isinstance(timer, _AlreadyDeleted):
            timer.stop()
        app._spacr_stall_timer = previous


@pytest.mark.slow
def test_the_stack_is_echoed_to_stderr_and_a_broken_stream_costs_nothing(
        tmp_path, monkeypatch, qapp):
    """Echo is the half a user sees without being told where the log is.

    It is also the half that is dangerous: this writes from a daemon thread,
    and a stream somebody else is swapping underneath it crashed pytest
    inside its own capture.py. So the write is guarded, and the guard is
    tested by giving it a stream that fails -- the file must still hold the
    report, because THE FILE IS THE RECORD AND STDERR IS A CONVENIENCE.

    The stream here is this test's own object, never pytest's.
    """
    import io
    import threading as _threading

    class _Stream(io.StringIO):
        def __init__(self):
            super().__init__()
            self.lock = _threading.Lock()
            self.fail = False

        def write(self, text):
            if self.fail:
                raise ValueError("I/O operation on closed file")
            with self.lock:
                return super().write(text)

    stream = _Stream()
    monkeypatch.setattr("sys.stderr", stream)
    working = tmp_path / "working"
    working.mkdir()
    text = _watch_one_stall(working, monkeypatch, echo=True)

    with stream.lock:
        echoed = stream.getvalue()
    assert "GUI THREAD STALLED" in echoed, "nothing reached stderr"
    assert "GUI THREAD STALLED" in text, "nothing reached the log either"

    # A SECOND STALL, INTO ITS OWN LOG. Reusing the first one would find the
    # marker already there and return without ever stalling, which would
    # make the assertion below true about nothing.
    stream.fail = True
    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    broken = _watch_one_stall(broken_dir, monkeypatch, echo=True)
    assert "GUI THREAD STALLED" in broken, (
        "a stderr that refused the write took the log entry with it")
    assert "WHERE THAT STALL SPENT ITS TIME" in broken, (
        "the summary went with it too")
