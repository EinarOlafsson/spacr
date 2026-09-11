"""Name the call that freezes the interface.

A STALLED GUI THREAD LEAVES NO TRACEBACK. It is not an exception and not a
fault, so ``faulthandler`` cannot see it, the log simply stops, and the
report that reaches the maintainer is "spaCR froze" with nothing under it.
The only thing that can name the call is a sample of the main thread's
stack taken WHILE it is stuck, from another thread.

    SPACR_WATCH_GUI_STALLS=1 spacr

Every time the event loop stops answering for longer than
:data:`STALL_SECONDS`, the exact Python stack of the GUI thread is written
to stderr and appended to :data:`LOG_PATH`. **The last frame of that stack
is the blocking call.**

WHY THIS IS IN THE PACKAGE RATHER THAN IN ``tools/``.
``tools/watch_the_gui_thread.py`` does the same thing and has to build the
``QApplication`` itself to attach before any screen exists -- which stopped
working the day :func:`spacr.qt.app.launch` began constructing its own, and
Qt refuses a second. Reaching in from outside was then tried two more ways
and failed twice more: hosting ``qt.run()`` inside another process makes
Home itself time out at thirty seconds, and a ``sitecustomize`` on the
startup benchmark worker's ``PYTHONPATH`` is imported but the
``QApplication`` it patches is never the one constructed. A flag read
inside ``launch`` is the one place that cannot be missed, and it composes
with every other driver -- the benchmark included, which is what this was
written for.

WHAT IT COSTS WHEN OFF: one ``os.environ.get``. When on: a 100 ms timer on
the GUI thread that increments an integer, and a daemon thread that
compares two integers four times a second.
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

LOG = logging.getLogger("spacr.qt.stall_watch")

#: How long the event loop may stop answering before it counts as a stall.
#: Well above a slow repaint and well below what a person calls "frozen".
STALL_SECONDS = float(os.environ.get("SPACR_STALL_SECONDS", "1.5"))

#: How often the watcher looks. Cheap: it compares two integers.
POLL_SECONDS = 0.25

#: Where the stacks are appended, so a user can send the file back.
LOG_PATH = Path(os.environ.get(
    "SPACR_STALL_LOG",
    str(Path.home() / ".spacr" / "logs" / "gui-stalls.log")))


def _write(text: str) -> None:
    """Append ``text`` to the stall log, and never raise for trying."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(text)
    except OSError:
        pass


def watch_this_application(app, *, stall_seconds: Optional[float] = None):
    """Report the GUI thread's stack whenever it stops answering.

    :param app: the live ``QApplication``. The heartbeat timer is parented
        to it so the timer lives exactly as long as the application does.
    :param stall_seconds: override :data:`STALL_SECONDS` for one call.
    :returns: the watcher thread, or ``None`` if Qt could not be reached.

    THE HEARTBEAT IS THE MEASUREMENT. A ``QTimer`` on the GUI thread bumps
    an integer; a daemon thread watches the integer. If it stops moving the
    GUI thread is not running the event loop, which is exactly the
    condition being hunted -- and the reason a timer cannot report it
    itself: a wedged loop does not deliver the timer either.
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
    # KEPT ON THE APPLICATION as well as parented to it: a local would be
    # collected the moment this function returns, and a collected QTimer
    # stops, which would leave the watcher reporting one endless stall.
    app._spacr_stall_timer = timer

    _write(f"\n=== watching the GUI thread (pid {os.getpid()}), "
           f"stall > {limit}s ===\n")

    def watch() -> None:
        """Sample the GUI thread through each stall and summarise it.

        ONE SNAPSHOT NAMES WHERE THE THREAD WAS, NOT WHERE THE TIME WENT,
        and those are different questions whenever the stall is a loop
        rather than a single blocking call. Measured: four stalls of the
        same module sweep gave four different last frames -- an event
        filter, a screen constructor, a settings-search install, another
        event filter -- which is a list of suspects rather than an answer.

        So the stack is sampled every :data:`POLL_SECONDS` FOR AS LONG AS
        the stall lasts, and the last frames are counted. A call that holds
        the thread appears in most samples; one that merely happened to be
        running appears in one.
        """
        reported_for = -1
        while True:
            time.sleep(POLL_SECONDS)
            stalled = time.monotonic() - beat["at"]
            if stalled < limit:
                continue
            if beat["n"] == reported_for:
                # SAME STALL, ANOTHER SAMPLE. The first crossing writes the
                # full stack; every later one only counts a frame, so a
                # thirteen-second freeze is one readable report rather than
                # fifty identical ones.
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
            try:
                sys.stderr.write(report)
                sys.stderr.flush()
            except Exception:                                # noqa: BLE001
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

    thread = threading.Thread(target=watch, daemon=True,
                              name="spacr-gui-stall-watch")
    thread.start()
    return thread
