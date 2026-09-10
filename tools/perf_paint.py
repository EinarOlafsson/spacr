#!/usr/bin/env python3
"""Measure what the startup benchmark does not: paint, theme and input.

Instruction 380 names five areas. `tools/spacr_startup_benchmark.py`
already covers three of them well -- cold and warm startup, per-module
time to ready, and Home -- with budgets, stall detection and a schema. It
does NOT measure the two that "smoothness" is actually judged by:

  * RUNNING THE THEMES. Paint rate of the animated backdrop, at 1080p and
    at 4K, per theme, and the cost of a whole-application theme change.
    Instruction 378 measured `app.setStyleSheet` + repolish at 587.5 ms
    on 847 widgets, and the one-backdrop fix found 950 paints/s and
    393 Mpx/s going into a layer nobody could see.
  * HANDLING THE WIDGETS AND SETTINGS. Input latency, which is not
    wall-clock: a 200 ms hitch on one keystroke is felt and 200 ms spread
    over ten frames is not.

WHY IT COUNTS PAINTS RATHER THAN GRABBING PIXELS. 380's own WATCH list:
`QWidget.grab()` does not capture the GL-backed `AmbientWidget`, so a
grab-based check reports a black window and cannot tell a working backdrop
from a broken one -- found the hard way after an A/B across two commits
came out byte-identical. `spacr.qt.widgets.ambient.total_frames_painted`
is the instrument the widget already carries, and it counts the thing that
costs.

CALIBRATED AGAINST A DEFECT THAT IS ALREADY FIXED, which is 380's first
instruction and the reason 350's sweep is believable: two backdrop layers
must read as roughly twice the paints of one. A harness that cannot see
the duplicated layer is not measuring paint, whatever number it prints.
See `tests/qt/test_the_paint_harness_can_see_a_second_layer.py`.

    python tools/perf_paint.py --seconds 3 --out docs/perf_baseline.json

A LOADED MACHINE MAKES EVERY NUMBER HERE SMALLER, so the JSON records the
load average and the platform beside the measurements. A baseline taken
while the test suite is running is a fact about that, not about spaCR.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The sizes the request names. 327 was reported from a 3840x2160 machine
#: and the cost is counted in pixels, so a 1080p-only measurement would
#: miss the case that was complained about by a factor of four.
SIZES = {"1080p": (1920, 1080), "4k": (3840, 2160)}

#: One frame at 60 Hz. An interaction that takes longer than this drops a
#: frame, which is what "smoothness" means when it is measured rather than
#: felt.
FRAME_MS = 1000.0 / 60.0

SCHEMA = 1


def _load() -> Dict[str, float]:
    """What else the machine was doing, so a number can be read honestly."""
    try:
        one, five, fifteen = os.getloadavg()
    except (AttributeError, OSError):                        # pragma: no cover
        return {}
    return {"load_1m": one, "load_5m": five, "load_15m": fifteen}


def _environment() -> dict:
    """Where this ran. A profile from one machine is a fact about it."""
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "qt_platform": os.environ.get("QT_QPA_PLATFORM", "(default)"),
        **_load(),
    }


def measure_backdrop(seconds: float = 3.0,
                     themes: Optional[List[str]] = None) -> List[dict]:
    """Paints per second for the ambient backdrop, per theme and size.

    :param seconds: how long to run the event loop for each combination.
    :param themes: theme names; None measures every theme spaCR offers.
    :returns: one row per (theme, size).
    """
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication

    from spacr.qt.preferences import PALETTE_THEMES
    from spacr.qt.widgets import ambient

    app = QApplication.instance() or QApplication([])
    # THE PALETTES, NOT `VALID_THEMES`, which also carries "system" -- a
    # name that means "whichever of these the desktop is set to" and would
    # measure one of the others twice under a label nobody could read.
    names = themes or list(PALETTE_THEMES)
    from spacr.qt.preferences import apply_preferences_to_app

    rows = []
    for name in names:
        # THE THEME HAS TO BE APPLIED BEFORE THE WIDGET IS BUILT, not
        # merely named in the row: the ambient look is read at
        # construction, and a bare `AmbientWidget()` measures the default
        # four times under four different labels. The first version of
        # this did exactly that, and the four identical numbers it printed
        # were the only clue.
        os.environ["SPACR_THEME"] = str(name)
        try:
            apply_preferences_to_app(app)
        except Exception:                                    # noqa: BLE001
            pass
        for label, (width, height) in SIZES.items():
            widget = ambient.AmbientWidget()
            widget.resize(width, height)
            widget.show()
            app.processEvents()
            before = ambient.total_frames_painted()
            started = time.perf_counter()
            loop = QEventLoop()
            QTimer.singleShot(int(seconds * 1000), loop.quit)
            loop.exec()
            elapsed = time.perf_counter() - started
            painted = ambient.total_frames_painted() - before
            _shut_down(widget, app)
            rows.append({
                "measurement": "backdrop",
                "theme": name,
                "size": label,
                "pixels": width * height,
                "seconds": round(elapsed, 3),
                "frames": painted,
                "fps": round(painted / elapsed, 1) if elapsed else 0.0,
                "megapixels_per_second": round(
                    painted * width * height / elapsed / 1e6, 1)
                if elapsed else 0.0,
            })
    return rows


def measure_theme_change(app_key: str = "mask") -> List[dict]:
    """What a whole-application theme change costs, with a widget count.

    378 measured 587.5 ms on 847 widgets against 107 ms for the visible
    screen alone. The widget count is recorded beside the time because the
    two only mean something together: the same milliseconds over twice the
    widgets is a different result.

    :param app_key: which module screen to build first, so the measurement
        is taken over a realistic widget tree rather than an empty window.
    :returns: one row per theme.
    """
    from PySide6.QtWidgets import QApplication, QWidget

    from spacr.qt.preferences import (PALETTE_THEMES,
                                      apply_preferences_to_app)
    from spacr.qt.screens.app_screen import AppScreen

    app = QApplication.instance() or QApplication([])
    screen = AppScreen(app_key=app_key)
    screen.resize(1600, 1000)
    screen.show()
    app.processEvents()
    # `allWidgets`, NOT `findChildren`: a top-level window is not a CHILD
    # of the application object, so the obvious call counts one widget on a
    # screen with eight hundred and reads as a flattering result.
    widgets = len(app.allWidgets())

    rows = []
    for name in PALETTE_THEMES:
        os.environ["SPACR_THEME"] = str(name)
        # The signature guard makes a repeated apply nearly free, which
        # would measure the guard rather than the work; force the sheet to
        # differ by clearing it first, the way a real theme change does.
        app.setStyleSheet("")
        started = time.perf_counter()
        try:
            apply_preferences_to_app(app)
        except Exception as error:                           # noqa: BLE001
            rows.append({"measurement": "theme", "theme": name,
                         "error": str(error)})
            continue
        app.processEvents()
        rows.append({
            "measurement": "theme",
            "theme": name,
            "widgets": widgets,
            "ms": round((time.perf_counter() - started) * 1000, 1),
            "screen": app_key,
        })
    _shut_down(screen, app)
    return rows


def measure_interaction(app_key: str = "mask") -> List[dict]:
    """Input latency for the interactions the request names.

    Expanding a section, typing a character, and scrolling the settings
    column -- each timed from the event being posted to the application
    being idle again, which is what a user waits through.

    :param app_key: the module screen to measure on.
    :returns: one row per interaction, with the frames it dropped.
    """
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QKeyEvent, QWheelEvent
    from PySide6.QtWidgets import (QApplication, QLineEdit, QScrollArea,
                                   QWidget)

    from spacr.qt.screens.app_screen import AppScreen

    app = QApplication.instance() or QApplication([])
    screen = AppScreen(app_key=app_key)
    screen.resize(1600, 1000)
    screen.show()
    app.processEvents()

    rows: List[dict] = []

    def timed(name: str, action, repeats: int = 3) -> None:
        """Time an interaction cold and then warm.

        BOTH NUMBERS, because they are different questions and the first
        one is the one a user meets. Typing the first character into a
        freshly built Mask panel measured 39 ms -- two dropped frames --
        and the second character 0.1. Recording only the repeat would have
        reported a keystroke as free; recording only the first would blame
        every later keystroke for work done once.
        """
        taken = []
        for _ in range(max(1, repeats)):
            started = time.perf_counter()
            try:
                action()
            except Exception as error:                       # noqa: BLE001
                rows.append({"measurement": "interaction", "action": name,
                             "error": str(error)})
                return
            app.processEvents()
            taken.append((time.perf_counter() - started) * 1000)
        first, rest = taken[0], taken[1:]
        rows.append({
            "measurement": "interaction",
            "action": name,
            "screen": app_key,
            "first_ms": round(first, 2),
            "repeat_ms": round(min(rest), 2) if rest else round(first, 2),
            "frames_dropped": max(0, int(first // FRAME_MS)),
        })

    # BY THE CLASSES THE PANEL ACTUALLY USES, not by a duck-typed
    # `hasattr`: the first version of this looked for anything carrying
    # `set_expanded` and found nothing at all, which reads exactly like
    # "expanding a section is free".
    from spacr.qt.widgets.collapsible_section import CollapsibleSection
    from spacr.qt.widgets.section import Section

    sections = (screen.findChildren(Section)
                + screen.findChildren(CollapsibleSection))
    if sections:
        # ALTERNATED, so the repeat is a real repeat: expanding a section
        # that is already expanded measures a no-op branch.
        timed("expand a section",
              lambda: (sections[0].set_expanded(False),
                       sections[0].set_expanded(True)))
        timed("collapse a section",
              lambda: (sections[0].set_expanded(True),
                       sections[0].set_expanded(False)))

    fields = screen.findChildren(QLineEdit)
    if fields:
        field = fields[0]
        field.setFocus()

        def type_one():
            event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key_A,
                              Qt.NoModifier, "a")
            QApplication.sendEvent(field, event)

        timed("type one character", type_one)

    areas = screen.findChildren(QScrollArea)
    if areas:
        area = areas[0]

        def scroll_once():
            bar = area.verticalScrollBar()
            bar.setValue(min(bar.maximum(), bar.value() + 240))

        timed("scroll the settings column", scroll_once)

    _shut_down(screen, app)
    return rows


def _shut_down(screen, app) -> None:
    """Close a measured screen without taking the process with it.

    `deleteLater` alone is not enough and the harness dumped core proving
    it: a module screen owns worker threads and preview timers, and
    dropping the widget while one is running gives "QThread: Destroyed
    while thread is still running" and a crash AFTER the numbers have
    been printed -- which looks like a measurement failure and is not.
    Closing first lets the screen stop what it started.
    """
    try:
        screen.close()
    except Exception:                                        # noqa: BLE001
        pass
    app.processEvents()
    try:
        screen.deleteLater()
    except Exception:                                        # noqa: BLE001
        pass
    app.processEvents()


def main(argv: Optional[List[str]] = None) -> int:
    """Run the measurements and print or write the record."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=3.0,
                        help="how long each backdrop measurement runs")
    parser.add_argument("--screen", default="mask",
                        help="which module screen to measure on")
    parser.add_argument("--only", choices=("backdrop", "theme", "interaction"),
                        help="run one measurement instead of all three")
    parser.add_argument("--out", type=Path,
                        help="write the record here as JSON")
    args = parser.parse_args(argv)

    rows: List[dict] = []
    if args.only in (None, "backdrop"):
        rows.extend(measure_backdrop(args.seconds))
    if args.only in (None, "theme"):
        rows.extend(measure_theme_change(args.screen))
    if args.only in (None, "interaction"):
        rows.extend(measure_interaction(args.screen))

    record = {
        "schema": SCHEMA,
        "environment": _environment(),
        "measurements": rows,
    }
    text = json.dumps(record, indent=1, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"written: {args.out}")
    for row in rows:
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items()
                               if k != "measurement"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
