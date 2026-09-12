#!/usr/bin/env python
"""Run spaCR and print the stack of the GUI thread whenever it stalls.

WHY. On 2026-09-04 the maintainer reported "opening map barcodes crashes
spacr". It is not a crash: `os.path.exists` on a path under `/nas_mnt` -- an
``autofs`` mount whose share was asleep -- did not return after twenty
seconds, and a blocked GUI thread has no traceback to leave behind. The same
stall also explains the hover flicker (events queue, then replay in a burst),
the deferred repaints, and the glimpses of other screens.

One such call was fixed. If the freeze survives, there is another, and
guessing where is what this exists to stop.

    python tools/watch_the_gui_thread.py

Use spaCR normally and make it freeze. Every time the interface stops
answering for longer than :data:`STALL_SECONDS`, the exact Python stack of
the GUI thread is written to stderr and appended to

    ~/.spacr/logs/gui-stalls.log

The last frame of that stack IS the blocking call. Send that file back.

HOW IT KNOWS. A `QTimer` on the GUI thread bumps a counter; a daemon thread
watches the counter. If it stops moving, the GUI thread is not running the
event loop, which is exactly the condition being hunted. `faulthandler`
cannot do this on its own -- it reports on a signal or a fault, and a stall
is neither.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from pathlib import Path

#: How long the event loop may stop answering before it counts as a stall.
#: Well above a slow repaint and well below what a person calls "frozen".
STALL_SECONDS = float(os.environ.get("SPACR_STALL_SECONDS", "1.5"))

#: How often the watchdog looks. Cheap: it compares two integers.
POLL_SECONDS = 0.25

LOG = Path.home() / ".spacr" / "logs" / "gui-stalls.log"


def _install(app) -> None:
    """Start the heartbeat on the GUI thread and the watcher beside it."""
    from PySide6.QtCore import QTimer

    beat = {"n": 0, "at": time.monotonic()}
    main_thread = threading.main_thread()

    def tick() -> None:
        beat["n"] += 1
        beat["at"] = time.monotonic()

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(100)
    app._stall_timer = timer          # keep it alive

    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== watching (pid {os.getpid()}), "
                     f"stall > {STALL_SECONDS}s ===\n")

    def watch() -> None:
        reported_for = -1
        while True:
            time.sleep(POLL_SECONDS)
            stalled = time.monotonic() - beat["at"]
            if stalled < STALL_SECONDS:
                continue
            if beat["n"] == reported_for:
                continue              # already reported THIS stall
            reported_for = beat["n"]

            frame = sys._current_frames().get(main_thread.ident)
            if frame is None:
                continue
            stack = "".join(traceback.format_stack(frame))
            report = (
                f"\n--- GUI THREAD STALLED {stalled:.1f}s "
                f"(heartbeat {beat['n']}) ---\n{stack}"
            )
            sys.stderr.write(report)
            sys.stderr.flush()
            try:
                with LOG.open("a", encoding="utf-8") as handle:
                    handle.write(report)
            except OSError:
                pass

    threading.Thread(target=watch, daemon=True,
                     name="spacr-gui-stall-watchdog").start()


def main() -> int:
    """Attach the watchdog to the application spaCR builds, then launch.

    THIS TOOL COULD NOT START, and the diagnostic being unavailable when it
    is wanted is the worst time for that. It used to make a `QApplication`
    of its own first, "so the watchdog can attach before any screen is
    built". `spacr.qt.app.launch` now constructs one unconditionally, and
    Qt refuses the second:

        RuntimeError: libshiboken: Please destroy the QApplication
        singleton before creating a new QApplication instance

    So the tool raised that before printing anything, on every invocation,
    including `--help`. Found 2026-09-11 while pointing it at the startup
    benchmark.

    The fix keeps the property the old comment wanted -- attached before
    any screen exists -- by wrapping the class `launch` constructs rather
    than racing it to construct one first. `_install` runs inside
    `__init__`, which is earlier than anything a pre-made instance could
    have achieved.

    SYS.PATH, TOO. A script's `sys.path[0]` is its own directory, so
    `tools/watch_the_gui_thread.py` imported whatever spaCR was INSTALLED
    rather than the checkout it lives in -- and then reported on a tree
    nobody was editing. The repository root goes first.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from PySide6.QtWidgets import QApplication

    from spacr.qt import app as app_module

    class _WatchedApplication(QApplication):
        """A ``QApplication`` that starts the stall watchdog on itself."""

        def __init__(self, *args, **kwargs):
            """Build the application and attach the watchdog to it."""
            super().__init__(*args, **kwargs)
            _install(self)

    existing = QApplication.instance()
    if existing is not None:
        # Someone already made one -- attach to it and leave the class
        # alone, because `launch` will reuse the singleton.
        _install(existing)
    else:
        app_module.QApplication = _WatchedApplication

    print(f"watching the GUI thread; stalls over {STALL_SECONDS}s go to\n"
          f"  {LOG}\n"
          "Open the module that freezes, then send that file back.",
          file=sys.stderr)

    from spacr.qt import run
    return run() or 0


if __name__ == "__main__":
    raise SystemExit(main())
