#!/usr/bin/env python3
"""Measure what an IDLE spaCR window repaints, per screen, per widget class.

Item 728. The native Mac run of 2026-10-25 found an idle spaCR at 90 % of a
core on Home and 110 % on Mask, because every translucent settings row was
repainted on every frame of the animated backdrop: 154 widget paints per
backdrop frame on Home, 196 on Mask and Measure. This is the instrument that
turns that into a number on any machine, so the fix can be measured before
and after rather than judged by eye.

It builds the real MainWindow (settings in a throwaway folder, so the
machine measured is not edited, with the first-run tour and the module
walkthroughs marked seen, since their dimming overlay is not an idle
window), opens each screen, lets it settle, and then
for ``--seconds`` counts:

  * backdrop frames: paints of the animated backdrop widget itself;
  * widget paints: every other paint event the window delivers, by class;
  * process CPU: user + system time of the whole process (GUI thread and
    shading thread together) over wall time, as a percentage of one core,
    and split into the GUI thread and every other thread. It is taken in
    its own window before the paints are counted, because the counting
    filter is Python and would add its own cost to every paint.

``--backdrop off`` is the control row: the same screens with the animated
backdrop switched off in Preferences.

Run it offscreen (``QT_QPA_PLATFORM=offscreen``) or on a real display. Under
Xvfb without a window manager the window is never exposed, the backdrop does
not tick, and every row reads zero.

    QT_QPA_PLATFORM=offscreen \\
        python tools/measure_idle_repaint.py --backdrop on --out on.json
    python tools/measure_idle_repaint.py --backdrop off --screens home

A loaded machine slows the backdrop and raises the CPU share of everything,
so the JSON records the load average beside the measurements. Paints per
backdrop frame does not depend on the load, and is the figure to compare
across machines.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

#: Screens the Mac report measured, and the key MainWindow opens each by.
SCREENS = {"home": "__home__", "mask": "mask", "measure": "measure"}


def _settle(app, seconds: float) -> None:
    """Run the event loop for ``seconds`` of wall time.

    A real ``QEventLoop`` rather than a ``processEvents`` poll, so the
    measuring itself costs the process nothing while it waits.
    """
    from PySide6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    QTimer.singleShot(int(seconds * 1000), loop.quit)
    loop.exec()


def _thread_times(process, gui_id: int):
    """CPU seconds of the GUI thread and of every other thread."""
    gui = other = 0.0
    for thread in process.threads():
        spent = thread.user_time + thread.system_time
        if thread.id == gui_id:
            gui += spent
        else:
            other += spent
    return gui, other


class _PaintCounter:
    """An application event filter that counts paint events by class."""

    def __init__(self):
        """Start with nothing counted."""
        from PySide6.QtCore import QObject

        counter = self

        class _Filter(QObject):
            """Count, never consume."""

            def eventFilter(self, watched, event):
                """Count a paint event by its widget's class."""
                if event.type() == event.Type.Paint and counter.on:
                    counter.counts[type(watched).__name__] += 1
                return False

        self.counts = Counter()
        self.on = False
        self.filter = _Filter()


def measure(screens, seconds: float, settle: float, backdrop: bool,
            theme: str, size, warmup: float = 25.0) -> dict:
    """Measure each of ``screens``; see the module docstring."""
    import psutil
    from PySide6.QtWidgets import QApplication

    from perf_paint import _environment, _settings_elsewhere

    app = QApplication.instance() or QApplication([])
    rows = []
    with _settings_elsewhere():
        from spacr.qt import preferences, register_self_registering_modules
        from spacr.qt.app import MainWindow
        from spacr.qt.widgets import ambient

        from spacr.qt import first_run, walkthrough

        first_run.mark_tour_seen()
        for seen in ("home", "__home__", *SCREENS.values()):
            walkthrough.mark_seen(seen)
        preferences.set_theme(theme)
        preferences.set_ambient_enabled(bool(backdrop))
        preferences.apply_preferences_to_app(app)
        register_self_registering_modules()
        window = MainWindow()
        window.resize(*size)
        window.show()
        _settle(app, warmup)
        counter = _PaintCounter()
        app.installEventFilter(counter.filter)
        process = psutil.Process()
        gui_id = threading.get_native_id()
        for name in screens:
            key = SCREENS[name]
            if key == "__home__":
                window._on_nav_selected(key)
            else:
                window.open_module(key)
            _settle(app, settle)
            cpu_before = process.cpu_times()
            threads_before = _thread_times(process, gui_id)
            cpu_wall = time.perf_counter()
            _settle(app, seconds)
            cpu_wall = time.perf_counter() - cpu_wall
            cpu_after = process.cpu_times()
            threads_after = _thread_times(process, gui_id)
            cpu = ((cpu_after.user - cpu_before.user)
                   + (cpu_after.system - cpu_before.system))
            counter.counts.clear()
            frames_before = ambient.total_frames_painted()
            wall_before = time.perf_counter()
            counter.on = True
            _settle(app, seconds)
            counter.on = False
            wall = time.perf_counter() - wall_before
            frames = ambient.total_frames_painted() - frames_before
            by_class = dict(counter.counts)
            by_class.pop("AmbientWidget", None)
            widget_paints = sum(by_class.values())
            rows.append({
                "screen": name,
                "backdrop": "on" if backdrop else "off",
                "theme": theme,
                "seconds": round(wall, 2),
                "backdrop_fps": round(frames / wall, 1),
                "widget_paints_per_s": round(widget_paints / wall, 1),
                "widget_paints_per_frame": (round(widget_paints / frames, 1)
                                            if frames else None),
                "cpu_percent_of_a_core": round(100.0 * cpu / cpu_wall, 1),
                "gui_thread_cpu_percent": round(
                    100.0 * (threads_after[0] - threads_before[0]) / cpu_wall, 1),
                "other_threads_cpu_percent": round(
                    100.0 * (threads_after[1] - threads_before[1]) / cpu_wall, 1),
                "by_class_per_s": {
                    cls: round(count / wall, 1) for cls, count in
                    sorted(by_class.items(), key=lambda item: -item[1])},
            })
        app.removeEventFilter(counter.filter)
        window.close()
        app.processEvents()
    return {"schema": 1, "environment": _environment(), "rows": rows}


def main(argv=None) -> int:
    """Parse the arguments, measure, print a table and write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--screens", default="home,mask,measure",
                        help="comma-separated: " + ", ".join(SCREENS))
    parser.add_argument("--backdrop", choices=("on", "off"), default="on")
    parser.add_argument("--theme", default="dark")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--settle", type=float, default=6.0)
    parser.add_argument("--size", default="1600x1000")
    parser.add_argument("--warmup", type=float, default=25.0,
                        help="seconds after the window shows before the "
                             "first screen is measured: the preloader and "
                             "the backdrop kernel compile run on other "
                             "threads first, and are not the idle cost")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    screens = [name.strip() for name in args.screens.split(",") if name.strip()]
    width, height = (int(part) for part in args.size.lower().split("x"))
    result = measure(screens, args.seconds, args.settle,
                     args.backdrop == "on", args.theme, (width, height),
                     args.warmup)
    for row in result["rows"]:
        top = ", ".join(f"{cls} {rate:.0f}" for cls, rate in
                        list(row["by_class_per_s"].items())[:6])
        print(f"{row['screen']:8s} backdrop {row['backdrop']:3s} "
              f"{row['backdrop_fps']:5.1f} fps  "
              f"{row['widget_paints_per_s']:8.1f} paints/s  "
              f"{row['widget_paints_per_frame']} per frame  "
              f"CPU {row['cpu_percent_of_a_core']:5.1f} % "
              f"(GUI {row['gui_thread_cpu_percent']:.1f}, "
              f"other {row['other_threads_cpu_percent']:.1f})  [{top}]")
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
