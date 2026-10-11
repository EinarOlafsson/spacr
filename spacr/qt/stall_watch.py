"""Name the call that freezes the interface.

A stalled GUI thread leaves no traceback. It is not an exception and not a
fault, so ``faulthandler`` does not report it and the log stops. Only a sample
of the main thread's stack, taken from another thread while it is stuck, names
the call.

    SPACR_WATCH_GUI_STALLS=1 spacr

Every time the event loop stops answering for longer than
:data:`STALL_SECONDS`, the exact Python stack of the GUI thread is written
to stderr and appended to :data:`LOG_PATH`. **The last frame of that stack
is the blocking call.**

This lives in the package rather than in ``tools/``.
``tools/watch_the_gui_thread.py`` does the same thing but has to build the
``QApplication`` itself to attach before any screen exists, which no longer
works because :func:`spacr.qt.app.launch` constructs its own and Qt refuses a
second.

Two other ways of attaching from outside do not work either. Hosting
``qt.run()`` inside another process makes Home itself time out at thirty
seconds.

The other way was a ``sitecustomize`` on the startup benchmark path. It is
imported, and what it changes is never the object that runs.

A flag read inside ``launch`` is the one place that cannot be missed, and it
composes with every other driver -- the benchmark included, which is what
this was written for.

Cost when off: one ``os.environ.get``. When on: a 100 ms timer on the GUI
thread that increments an integer, and a daemon thread that compares two
integers four times a second.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Optional
from ..logging_util import _spacr_home

LOG = logging.getLogger("spacr.qt.stall_watch")

#: How long the event loop may stop answering before it counts as a stall.
#: Well above a slow repaint and well below what a person calls "frozen".
STALL_SECONDS = float(os.environ.get("SPACR_STALL_SECONDS", "1.5"))

#: How often the watcher looks. Cheap: it compares two integers.
POLL_SECONDS = 0.25

#: Where the stacks are appended, so a user can send the file back.
LOG_PATH = Path(os.environ.get(
    "SPACR_STALL_LOG",
    str(_spacr_home() / "logs" / "gui-stalls.log")))


def _write(text: str) -> None:
    """Append ``text`` to the stall log, and never raise for trying."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(text)
    except OSError:
        pass


def watch_this_application(app, *, stall_seconds: Optional[float] = None,
                           echo: bool = True):
    """Report the GUI thread's stack whenever it stops answering.

    :param app: the live ``QApplication``. The heartbeat timer is parented
        to it so the timer lives exactly as long as the application does.
    :param stall_seconds: override :data:`STALL_SECONDS` for one call.
    :param echo: also write each report to ``sys.stderr``. Pass ``False`` under
        a harness that captures streams. The file is the record and stderr is
        optional: this writes from a daemon thread, and writing into a stream
        that a test harness is replacing crashed pytest inside its own
        `capture.py`.
    :returns: the watcher thread, or ``None`` if Qt could not be reached.

    The heartbeat is the measurement. A ``QTimer`` on the GUI thread bumps an
    integer; a daemon thread watches the integer. If it stops moving the GUI
    thread is not running the event loop, which is exactly the condition being
    hunted -- and the reason a timer cannot report it itself: a wedged loop
    does not deliver the timer either.
    """
    try:
        from PySide6.QtCore import QTimer
    except Exception:                                        # noqa: BLE001
        LOG.debug("no Qt, so no stall watch", exc_info=True)
        return None

    limit = float(STALL_SECONDS if stall_seconds is None else stall_seconds)
    beat = {"n": 0, "at": time.monotonic()}
    main_thread = threading.main_thread()

    def tick() -> None:
        """Record that the event loop is still turning."""
        beat["n"] += 1
        beat["at"] = time.monotonic()

    timer = QTimer(app)
    timer.timeout.connect(tick)
    timer.start(100)
    previous = getattr(app, "_spacr_stall_timer", None)
    if previous is not None:
        try:
            previous.stop()
        except Exception:                                    # noqa: BLE001
            pass
    app._spacr_stall_timer = timer

    _write(f"\n=== watching the GUI thread (pid {os.getpid()}), "
           f"stall > {limit}s ===\n")

    stopping = threading.Event()

    def watch() -> None:
        """Sample the GUI thread through each stall and summarise it.

        One snapshot shows where the thread was, not where the time went; the
        two differ when the stall is a loop rather than a single blocking call.
        Measured: four stalls of the same module sweep gave four different last
        frames -- an event filter, a screen constructor, a settings-search
        install, another event filter -- which is a list of suspects rather
        than an answer.

        So the stack is sampled every :data:`POLL_SECONDS` for as long as the
        stall lasts, and the last frames are counted. A call that holds the
        thread appears in most samples; one that merely happened to be running
        appears in one.
        """
        reported_for = -1
        while not stopping.is_set():
            stopping.wait(POLL_SECONDS)
            if stopping.is_set():
                break
            stalled = time.monotonic() - beat["at"]
            if stalled < limit:
                if samples:
                    _flush_samples()
                continue
            if beat["n"] == reported_for:
                frame = sys._current_frames().get(main_thread.ident)
                if frame is not None:
                    samples.append(_where(frame))
                continue
            _flush_samples()
            reported_for = beat["n"]
            frame = sys._current_frames().get(main_thread.ident)
            if frame is None:
                continue
            samples.append(_where(frame))
            report = (f"\n--- GUI THREAD STALLED {stalled:.1f}s "
                      f"(heartbeat {beat['n']}) ---\n"
                      + "".join(traceback.format_stack(frame)))
            if echo:
                try:
                    sys.stderr.write(report)
                    sys.stderr.flush()
                except Exception:                            # noqa: BLE001
                    pass
            _write(report)

    samples: list = []

    def _where(frame) -> str:
        """The innermost frame, as ``file:line function``."""
        stack = traceback.extract_stack(frame)
        if not stack:
            return "(empty)"
        last = stack[-1]
        name = os.path.basename(last.filename)
        return f"{name}:{last.lineno} {last.name}"

    def _flush_samples() -> None:
        """Write what held the thread through the stall that just ended."""
        if len(samples) < 2:
            samples.clear()
            return
        counts: dict = {}
        for where in samples:
            counts[where] = counts.get(where, 0) + 1
        total = len(samples)
        ranked = sorted(counts.items(), key=lambda row: -row[1])[:8]
        lines = "".join(
            f"    {count:4d}/{total}  {100.0 * count / total:5.1f}%  {where}\n"
            for where, count in ranked)
        _write(f"\n    WHERE THAT STALL SPENT ITS TIME "
               f"({total} samples at {POLL_SECONDS}s):\n{lines}")
        samples.clear()

    def _flush_when_the_stall_ends() -> None:
        """Summarise a stall the moment the loop answers again.

        Summarising when the stall ends, rather than when the next stall
        begins, means the last stall of a session is summarised even if the
        process is then killed, for example after a hang on a sleeping autofs
        mount. The summary separates the call holding the thread from one that
        happened to be running when a sample was taken.

        The heartbeat resuming is the end of the stall, so that is where the
        summary belongs. `tick` cannot do it, because it runs on the GUI
        thread, which was not running during the stall.
        """
        if samples:
            _flush_samples()

    def watch_and_flush() -> None:
        """Run the watcher, and summarise whatever is pending on the way out."""
        try:
            watch()
        finally:                                             # pragma: no cover
            _flush_when_the_stall_ends()

    thread = threading.Thread(target=watch_and_flush, daemon=True,
                              name="spacr-gui-stall-watch")
    thread.stop = stopping.set
    thread.start()
    return thread
