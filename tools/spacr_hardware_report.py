#!/usr/bin/env python3
"""Measure spaCR startup and interface performance on the current machine.

Run this command on a machine where spaCR is slow:

    python tools/spacr_hardware_report.py

The command prints the report and saves a copy under ``~/.spacr/reports``;
the final line gives the path. ``--out PATH`` selects another destination,
and ``--quick`` skips the longer benchmarks.

The report does not open a project or read project data. Because it follows
the normal GUI startup path, spaCR may update its usual local logs and caches
in addition to writing the report.

The measurements cover the real startup path, display scaling, an actual
paint, and animation loops. Each section handles errors independently so that
the remaining measurements are still reported when one component is
unavailable.
"""
from __future__ import annotations

import contextlib
import io
import os
import platform
import sys
import time
import traceback

#: Sections that take real time, skipped by --quick.
SLOW = "SLOW"


class Report:
    """Collects the report while printing it."""

    _out = staticmethod(print)

    def __init__(self) -> None:
        self.lines: list = []

    def __call__(self, text: str = "") -> None:
        Report._out(text)
        self.lines.append(str(text))

    def rule(self, title: str) -> None:
        self("")
        self(f"-- {title} " + "-" * max(0, 66 - len(title)))

    def item(self, name: str, value) -> None:
        self(f"  {name:<32} {value}")

    def timed(self, name: str, seconds: float, note: str = "") -> None:
        self(f"  {name:<32} {seconds:8.3f} s  {note}")

    def failed(self, name: str, exc: BaseException) -> None:
        self(f"  {name:<32} FAILED  {type(exc).__name__}: {exc}")


def _clock():
    return time.perf_counter()


@contextlib.contextmanager
def _section(say: Report, title: str):
    say.rule(title)
    try:
        yield
    except BaseException as exc:                             # noqa: BLE001
        say(f"  section failed: {type(exc).__name__}: {exc}")
        for line in traceback.format_exc().splitlines()[-4:]:
            say(f"    {line}")


# ---------------------------------------------------------------- machine

def machine(say: Report) -> None:
    with _section(say, "the machine"):
        say.item("python", sys.version.split()[0])
        say.item("executable", sys.executable)
        say.item("platform", platform.platform())
        say.item("machine", platform.machine())
        say.item("processor", platform.processor() or "(not reported)")
        try:
            import multiprocessing
            say.item("cpu count", multiprocessing.cpu_count())
        except Exception:
            pass
        try:
            import psutil
            mem = psutil.virtual_memory()
            say.item("memory total", f"{mem.total / 2**30:.1f} GiB")
            say.item("memory available", f"{mem.available / 2**30:.1f} GiB")
            freq = psutil.cpu_freq()
            if freq:
                say.item("cpu freq", f"{freq.current:.0f} MHz")
            say.item("cpu load now", f"{psutil.cpu_percent(interval=0.5):.0f} %")
        except Exception as exc:                             # noqa: BLE001
            say.item("psutil", f"unavailable ({type(exc).__name__})")

        # EMULATION IS INVISIBLE UNLESS ASKED, and it is the single biggest
        # multiplier there is: an x86_64 Python on Apple Silicon is emulated.
        if platform.system() == "Darwin":
            for name, args in (
                ("translated (Rosetta)", ["sysctl", "-n", "sysctl.proc_translated"]),
                ("cpu brand", ["sysctl", "-n", "machdep.cpu.brand_string"]),
                ("performance cores", ["sysctl", "-n", "hw.perflevel0.logicalcpu"]),
                ("efficiency cores", ["sysctl", "-n", "hw.perflevel1.logicalcpu"]),
                ("thermal pressure", ["pmset", "-g", "therm"]),
            ):
                try:
                    import subprocess
                    out = subprocess.run(args, capture_output=True, text=True,
                                         timeout=8).stdout.strip()
                    if name.startswith("translated"):
                        out += " (RUNNING UNDER ROSETTA)" if out == "1" else " (native)"
                    say.item(name, out.replace("\n", " ")[:90] or "(empty)")
                except Exception as exc:                     # noqa: BLE001
                    say.item(name, f"could not ask ({type(exc).__name__})")


# ---------------------------------------------------------------- imports

def imports(say: Report) -> None:
    with _section(say, "imports"):
        total = 0.0
        for name in ("numpy", "pandas", "scipy", "PySide6.QtCore",
                     "PySide6.QtGui", "PySide6.QtWidgets", "matplotlib",
                     "pyqtgraph", "torch", "sklearn", "statsmodels",
                     "cellpose", "umap", "numba", "skimage", "spacr"):
            start = _clock()
            try:
                __import__(name)
                took = _clock() - start
                total += took
                say.timed(name, took)
            except Exception as exc:                         # noqa: BLE001
                say.item(name, f"not importable ({type(exc).__name__})")
        say.timed("TOTAL", total)


def numerics(say: Report, quick: bool) -> None:
    with _section(say, "the numeric stack"):
        import numpy as np
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                np.__config__.show()
            text = buf.getvalue().lower()
            found = [n for n in ("accelerate", "openblas", "mkl", "blis")
                     if n in text]
            say.item("numpy BLAS", ", ".join(found) or "(not reported)")
        except Exception:
            say.item("numpy BLAS", "(could not ask)")
        say.item("numpy threads", os.environ.get("OMP_NUM_THREADS", "(unset)"))

        for label, size in (("matmul 1200", 1200), ("matmul 2400", 2400)):
            if quick and size > 1200:
                continue
            try:
                a = np.random.rand(size, size)
                start = _clock(); a @ a
                say.timed(label, _clock() - start)
            except Exception as exc:                         # noqa: BLE001
                say.failed(label, exc)
        try:
            import pandas as pd
            frame = pd.DataFrame(np.random.rand(200_000, 8))
            start = _clock(); frame.groupby(frame[0] > 0.5).mean()
            say.timed("pandas groupby 200k", _clock() - start)
        except Exception as exc:                             # noqa: BLE001
            say.failed("pandas groupby", exc)
        try:
            import torch
            say.item("torch version", torch.__version__)
            say.item("torch cuda", torch.cuda.is_available())
            say.item("torch mps",
                     getattr(torch.backends, "mps", None) is not None
                     and torch.backends.mps.is_available())
            t = torch.rand(1500, 1500)
            start = _clock(); t @ t
            say.timed("torch matmul 1500 cpu", _clock() - start)
        except Exception as exc:                             # noqa: BLE001
            say.failed("torch", exc)


# ------------------------------------------------------------------ qt

def _app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def display(say: Report) -> None:
    with _section(say, "the display -- WHERE HIDPI SHOWS UP"):
        from PySide6.QtCore import qVersion
        say.item("Qt version", qVersion())
        app = _app()
        say.item("platform plugin", app.platformName())
        for index, screen in enumerate(app.screens()):
            geo = screen.geometry()
            say.item(f"screen {index} name", screen.name())
            say.item(f"screen {index} logical size", f"{geo.width()} x {geo.height()}")
            ratio = screen.devicePixelRatio()
            say.item(f"screen {index} devicePixelRatio", ratio)
            say.item(f"screen {index} DEVICE pixels",
                     f"{int(geo.width() * ratio)} x {int(geo.height() * ratio)}"
                     f"  ({int(geo.width() * geo.height() * ratio * ratio):,} px)")
            say.item(f"screen {index} refresh", f"{screen.refreshRate():.0f} Hz")
            say.item(f"screen {index} logical DPI", f"{screen.logicalDotsPerInch():.0f}")


def preferences_in_effect(say: Report) -> None:
    with _section(say, "the preferences actually in effect"):
        _app()
        import spacr.qt.preferences as prefs
        for label, getter in (
            ("zoom / font scale", "get_font_scale"),
            ("theme", "get_theme"),
            ("ambient enabled", "get_ambient_enabled"),
            ("ambient animation", "get_ambient_animation"),
            ("ambient theme", "get_ambient_theme"),
            ("ambient resolution", "get_ambient_resolution"),
            ("ambient density", "get_ambient_density"),
            ("ambient speed", "get_ambient_speed"),
            ("ambient blur", "get_ambient_blur"),
            ("spacr mode", "get_spacr_mode"),
            ("colourblind mode", "get_colourblind_mode"),
        ):
            fn = getattr(prefs, label and getter, None)
            if fn is None:
                continue
            try:
                say.item(label, fn())
            except Exception as exc:                         # noqa: BLE001
                say.item(label, f"could not read ({type(exc).__name__})")
        say.item("DEFAULT_FONT_SCALE", getattr(prefs, "DEFAULT_FONT_SCALE", "?"))


def theming(say: Report) -> None:
    with _section(say, "theme and stylesheet"):
        _app()
        from spacr.qt import theme
        start = _clock(); sheet = theme.stylesheet()
        say.timed("build stylesheet", _clock() - start,
                  f"{len(sheet):,} characters")
        start = _clock()
        for _ in range(20):
            theme.stylesheet()
        say.timed("20 more (cached?)", _clock() - start)


# ------------------------------------------------------- the application

#: Measured in a CHILD process, because by the time the report reaches this
#: point its own earlier sections have imported most of what the splash
#: would. A cold number needs a cold interpreter.
_SPLASH_PROBE = r"""
import json, sys, time
t0 = time.perf_counter()
from PySide6.QtWidgets import QApplication
app = QApplication([])
qt_ready = time.perf_counter() - t0
before = set(sys.modules)
t1 = time.perf_counter()
import spacr.qt as qt
import_qt = time.perf_counter() - t1
t2 = time.perf_counter()
qt.register_self_registering_modules()
registering = time.perf_counter() - t2
new = set(sys.modules) - before
roots = {}
for name in new:
    root = name.split(".")[0]
    roots[root] = roots.get(root, 0) + 1
t3 = time.perf_counter()
from spacr.qt.app import MainWindow
import_window = time.perf_counter() - t3
t4 = time.perf_counter()
window = MainWindow()
build_window = time.perf_counter() - t4
t5 = time.perf_counter()
app.processEvents()
first_events = time.perf_counter() - t5
print("SPLASH_JSON" + json.dumps({
    "qt_ready": qt_ready, "import_qt": import_qt,
    "registering": registering, "modules": len(new),
    "roots": sorted(roots.items(), key=lambda kv: -kv[1])[:10],
    "import_window": import_window, "build_window": build_window,
    "first_events": first_events,
    "total": time.perf_counter() - t0,
}))
"""


#: Times the REAL entry point -- what typing `spacr` runs -- by letting it
#: do everything up to the event loop and then returning from exec()
#: instead of blocking. Anything `launch()` does that a hand-built
#: MainWindow does not is inside this number and outside the one above.
_LAUNCH_PROBE = r"""
import json, sys, time
import PySide6.QtWidgets as W

marks = {}

def _stop(self, *a, **k):
    marks.setdefault("exec_reached", time.perf_counter())
    marks.setdefault("who", type(self).__name__)
    return 0

# Stop nested event loops as well as the application's main loop. A profile
# that requires setup reaches SetupSlides.exec() before QApplication.exec();
# without this replacement, the non-interactive probe would wait for input
# until its 900-second timeout expired.
W.QApplication.exec = _stop
W.QApplication.exec_ = _stop
W.QDialog.exec = _stop
W.QDialog.exec_ = _stop
try:
    import PySide6.QtCore as C
    C.QEventLoop.exec = _stop
    C.QEventLoop.exec_ = _stop
except Exception:
    pass

t0 = time.perf_counter()
import spacr.qt
code = spacr.qt.run([])
done = time.perf_counter()
reached = marks.get("exec_reached", done)
print("LAUNCH_JSON" + json.dumps({
    "to_event_loop": reached - t0,
    "total": done - t0,
    "exit": code,
    "modules": len(sys.modules),
    "first_loop": marks.get("who", "(none reached)"),
}))
"""


def the_real_launch(say: Report) -> None:
    """Measure the ``spacr`` entry point in a fresh interpreter.

    Unlike :func:`the_splash`, this measurement includes the work performed
    by :func:`spacr.qt.app.launch`, such as setup checks, loading saved state,
    and constructing the main window. The probe replaces Qt event loops so
    that it can return without requiring interaction, and records which loop
    the normal launch would reach first.
    """
    with _section(say, "the `spacr` entry point in a cold process"):
        import json
        import subprocess
        try:
            done = subprocess.run(
                [sys.executable, "-c", _LAUNCH_PROBE],
                capture_output=True, text=True, timeout=900,
                env={**os.environ})
        except Exception as exc:                             # noqa: BLE001
            say.failed("real launch probe", exc)
            return
        line = next((l for l in done.stdout.splitlines()
                     if l.startswith("LAUNCH_JSON")), "")
        if not line:
            say("  the launch probe produced no reading")
            for tail in (done.stderr or "").splitlines()[-6:]:
                say(f"    {tail}")
            return
        data = json.loads(line[len("LAUNCH_JSON"):])
        say.timed("run() to first event loop", data["to_event_loop"])
        say.timed("run() returned", data["total"])
        say.item("modules loaded by then", f"{data['modules']:,}")
        say.item("exit code", data["exit"])
        first_loop = str(data.get("first_loop", "?"))
        say.item("first event loop reached", first_loop)
        if first_loop == "SetupSlides":
            say("  The setup screen opened before the main application loop.")
            say("  A normal interactive launch waits here until setup is "
                "complete.")
            say("  For an unattended launch, use `spacr --no-setup`.")
        elif first_loop not in {"QApplication", "(none reached)", "?"}:
            say(f"  The {first_loop} event loop opened before the main "
                "application loop.")
            say("  Normal startup may wait for this event loop to finish.")
        say("")
        say("  If the return time is materially longer than the cold-launch")
        say("  total below, launch() is performing additional startup work.")


def the_splash(say: Report) -> None:
    """The phase between typing spacr and the window appearing.

    THIS IS THE ONE THE EARLIER REPORTS MISSED. They imported MainWindow
    directly, which skips `register_self_registering_modules` -- the step
    that imports every module owning an app so it can register itself. That
    is what the splash screen covers, and on a machine short of memory it is
    a thousand imports walked from a cold page cache.

    Run in a CHILD interpreter so nothing this report already imported can
    make the number look better than a real launch.
    """
    with _section(say, "THE SPLASH -- typing spacr until the window appears"):
        import json
        import subprocess
        try:
            done = subprocess.run(
                [sys.executable, "-c", _SPLASH_PROBE],
                capture_output=True, text=True, timeout=900,
                env={**os.environ, "QT_QPA_PLATFORM":
                     os.environ.get("QT_QPA_PLATFORM", "")})
        except Exception as exc:                             # noqa: BLE001
            say.failed("cold launch probe", exc)
            return
        line = next((l for l in done.stdout.splitlines()
                     if l.startswith("SPLASH_JSON")), "")
        if not line:
            say("  the cold probe produced no reading")
            for tail in (done.stderr or "").splitlines()[-4:]:
                say(f"    {tail}")
            return
        data = json.loads(line[len("SPLASH_JSON"):])
        say.timed("QApplication in a cold process", data["qt_ready"])
        say.timed("import spacr.qt", data["import_qt"])
        say.timed("REGISTER EVERY MODULE", data["registering"],
                  f"{data['modules']:,} modules imported")
        if data["modules"]:
            say.item("per module",
                     f"{data['registering'] / data['modules'] * 1000:.2f} ms")
        say.item("largest packages",
                 ", ".join(f"{n}({c})" for n, c in data["roots"]))
        say.timed("import MainWindow", data["import_window"])
        say.timed("build MainWindow", data["build_window"])
        say.timed("first processEvents", data["first_events"])
        say.timed("COLD LAUNCH TOTAL", data["total"],
                  "<-- this is the splash screen")


def startup(say: Report) -> None:
    """The real path: the window a launch actually builds."""
    with _section(say, "THE REAL STARTUP PATH"):
        app = _app()
        from spacr.qt.app import MainWindow
        start = _clock()
        window = MainWindow()
        built = _clock() - start
        say.timed("MainWindow()", built)
        start = _clock(); app.processEvents()
        say.timed("first processEvents", _clock() - start)
        start = _clock(); window.show(); app.processEvents()
        say.timed("show + events", _clock() - start)
        try:
            widgets = len(app.allWidgets())
            say.item("widgets alive", f"{widgets:,}")
        except Exception:
            pass
        start = _clock()
        for _ in range(10):
            app.processEvents()
        say.timed("10 idle event loops", _clock() - start)
        try:
            start = _clock(); window.grab()
            say.timed("grab one full paint", _clock() - start)
        except Exception as exc:                             # noqa: BLE001
            say.failed("grab", exc)
        window.close()


def screens(say: Report, quick: bool) -> None:
    title = "every live module: click to painted usable state"
    with _section(say, title + ("" if quick else f" [{SLOW}]")):
        if quick:
            say("  skipped by --quick; the complete registry sweep is never")
            say("  replaced with a hand-picked subset. Run without --quick or")
            say("  use tools/spacr_startup_benchmark.py directly.")
            return

        # A fresh child is essential here. Earlier report sections import the
        # scientific stack deliberately; measuring modules in this process
        # would call every heavy screen warm and understate the user path.
        import json
        import subprocess
        import tempfile
        from pathlib import Path

        driver = Path(__file__).with_name("spacr_startup_benchmark.py")
        with tempfile.TemporaryDirectory(prefix="spacr-hardware-modules-") as raw:
            output = Path(raw) / "modules.json"
            command = [
                sys.executable, str(driver), "--runs", "1", "--record-only",
                "--out", str(output), "--package-root",
                str(Path(__file__).resolve().parents[1]),
            ]
            if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
                command.append("--offscreen")
            done = subprocess.run(
                # The worker owns a 3600 s last-resort timeout and preserves
                # its last atomic checkpoint.  Give the driver one minute to
                # recover that evidence and write the combined artifact.
                command, capture_output=True, text=True, timeout=3660)
            if not output.is_file():
                say.item("registry benchmark", "FAILED (no JSON artifact)")
                for line in (done.stderr or "").splitlines()[-6:]:
                    say(f"    {line}")
                return
            artifact = json.loads(output.read_text(encoding="utf-8"))

        keys = artifact.get("registry_keys", [])
        say.item("live registry", f"{len(keys)} app(s)")
        run = (artifact.get("runs") or [{}])[0]
        benchmark = run.get("benchmark", {})
        measured = benchmark.get("measured_keys", [])
        say.item("measured registry", f"{len(measured)} app(s)")
        say.item("sets equal", measured == keys)
        for row in benchmark.get("results", []):
            detail = str(row.get("detail", "?"))
            seconds = float(row.get("duration_s", 0.0))
            stall = row.get("worst_event_loop_stall_ms")
            stall_text = "unknown" if stall is None else f"{float(stall):.0f} ms"
            error = f"  FAILED: {row['error']}" if row.get("error") else ""
            say.timed(detail, seconds, f"worst event gap {stall_text}{error}")
        for violation in benchmark.get("violations", []):
            say(f"  BUDGET VIOLATION: {violation}")


def backdrop(say: Report) -> None:
    with _section(say, "the animated backdrop"):
        _app()
        from spacr.qt import theme
        from spacr.qt.widgets import ambient
        page = theme.page_colour("dark")
        names = list(getattr(ambient, "AMBIENT_THEMES", ()) or ())
        extra = getattr(ambient, "SPACEOUT_THEME", None)
        if extra and extra not in names:
            names.append(extra)
        say.item("buffer max edge", getattr(ambient, "BUFFER_MAX_EDGE", "?"))
        say.item("buffer max pixels", f"{getattr(ambient, 'BUFFER_MAX_PIXELS', 0):,}")
        try:
            say.item("screen_pixels() sees", f"{ambient.screen_pixels():,}")
        except Exception:
            pass
        from PySide6.QtGui import QImage, QPainter

        def _draw(engine, canvas) -> None:
            """Paint one frame, however this engine spells it.

            The cost is in the SHADE, not in advancing the state, so a
            timing that never paints reports every engine as free. Engines
            differ in signature, so each shape is tried in turn rather than
            one difference being reported as a failure.
            """
            width, height = canvas.width(), canvas.height()
            shade = getattr(engine, "_shade", None)
            if callable(shade):
                try:
                    shade(width, height)
                    return
                except TypeError:
                    pass
            painter = QPainter(canvas)
            try:
                for attempt in (
                    lambda: engine.blit(painter, None, width, height),
                    lambda: engine.blit(painter, width, height),
                    lambda: engine.blit(painter, canvas.rect()),
                ):
                    try:
                        attempt()
                        return
                    except TypeError:
                        continue
            finally:
                painter.end()

        def _palette_for(theme_name: str) -> str:
            """A palette this theme accepts -- each offers a different set."""
            chooser = getattr(ambient, "default_palette_for", None)
            if callable(chooser):
                try:
                    return chooser(theme_name)
                except Exception:                            # noqa: BLE001
                    pass
            return getattr(ambient, "DEFAULT_PALETTE", "spacr")

        for name in names:
            try:
                engine = ambient.make_engine(
                    name, _palette_for(name), page, seed=1)
                # advance() only moves the state; the cost is in the SHADE,
                # so a timing that never paints reports every engine free.
                canvas = QImage(960, 540, QImage.Format_ARGB32_Premultiplied)
                for _ in range(3):
                    engine.advance(1 / 24)
                    _draw(engine, canvas)
                start = _clock()
                for _ in range(20):
                    engine.advance(1 / 24)
                    _draw(engine, canvas)
                ms = (_clock() - start) / 20 * 1000
                budget = 1000.0 / max(1, getattr(ambient, "DEFAULT_FPS", 24))
                flag = "  <-- OVER FRAME BUDGET" if ms > budget else ""
                say.item(f"{name}", f"{ms:7.2f} ms/frame (budget {budget:.1f}){flag}")
            except Exception as exc:                         # noqa: BLE001
                say.failed(name, exc)


def cursor_rim(say: Report) -> None:
    """The lit rim on the setup card, reported as laggy."""
    with _section(say, "the setup card's cursor rim"):
        app = _app()
        from spacr.qt.widgets.setup_card import SetupCard
        card = SetupCard()
        card.resize(900, 600)
        card.show()
        app.processEvents()
        start = _clock()
        for _ in range(30):
            if hasattr(card, "_aim_at_the_cursor"):
                card._aim_at_the_cursor()
            card.repaint()
            app.processEvents()
        ms = (_clock() - start) / 30 * 1000
        say.item("aim + repaint", f"{ms:7.2f} ms/frame"
                 + ("  <-- OVER 16 ms, this is the lag" if ms > 16 else ""))
        card.close()


def concurrency(say: Report) -> None:
    """What is running AT ONCE, and how long a queued event then waits.

    THE SECTIONS ABOVE MEASURE ONE THING AT A TIME, and that is not how the
    application runs. Ten things costing three milliseconds each are all
    comfortably inside a frame on their own and blow it together -- and on
    the GUI thread they do not share out, they QUEUE. What a user calls lag
    is the wait between an event arriving and being handled, so that wait is
    what this measures, with everything running.
    """
    with _section(say, "EVERYTHING AT ONCE -- what a user actually feels"):
        from PySide6.QtCore import QElapsedTimer, QObject, QTimer
        from PySide6.QtWidgets import QApplication
        app = _app()
        from spacr.qt.app import MainWindow

        window = MainWindow()
        window.resize(1600, 1000)
        window.show()
        app.processEvents()

        # WHAT IS TICKING. Every repeating timer on the GUI thread is a
        # slice of every frame, and they are easy to add and invisible
        # afterwards.
        timers = []
        for obj in window.findChildren(QObject):
            for child in obj.children():
                if isinstance(child, QTimer) and child.isActive():
                    timers.append(child.interval())
        for child in window.children():
            if isinstance(child, QTimer) and child.isActive():
                timers.append(child.interval())
        say.item("active repeating timers", len(timers))
        if timers:
            fastest = min(t for t in timers if t >= 0)
            say.item("fastest interval", f"{fastest} ms")
            per_second = sum(1000.0 / t for t in timers if t > 0)
            say.item("timer firings per second", f"{per_second:.0f}")

        try:
            import threading
            say.item("python threads alive", threading.active_count())
            say.item("thread names",
                     ", ".join(sorted(t.name for t in threading.enumerate()))[:80])
        except Exception:
            pass
        try:
            import psutil
            say.item("OS threads in process",
                     psutil.Process().num_threads())
        except Exception:
            pass

        # THE NUMBER THAT MATTERS. Post a zero-delay callback and see how
        # long it waits. Idle first, then with work queued behind it.
        def _latency(rounds: int = 40) -> float:
            worst = 0.0
            clock = QElapsedTimer()
            for _ in range(rounds):
                fired = []
                clock.start()
                QTimer.singleShot(0, lambda: fired.append(clock.nsecsElapsed()))
                spun = 0
                while not fired and spun < 200:
                    app.processEvents()
                    spun += 1
                if fired:
                    worst = max(worst, fired[0] / 1e6)
            return worst

        idle = _latency()
        say.item("event wait, idle", f"{idle:8.2f} ms")

        # Now with the backdrop painting every frame, which is the state the
        # application is actually in while a user is looking at it.
        try:
            from spacr.qt import theme
            from spacr.qt.widgets import ambient
            from PySide6.QtGui import QImage, QPainter
            name = ambient.default_palette_for("blobs")
            engine = ambient.make_engine("blobs", name,
                                         theme.page_colour("dark"), seed=1)
            canvas = QImage(1600, 1000, QImage.Format_ARGB32_Premultiplied)

            painting = QTimer()
            painting.setInterval(int(1000 / max(1, getattr(
                ambient, "DEFAULT_FPS", 24))))

            def _tick():
                engine.advance(painting.interval() / 1000.0)
                try:
                    engine._shade(canvas.width(), canvas.height())
                except Exception:                            # noqa: BLE001
                    pass

            painting.timeout.connect(_tick)
            painting.start()
            busy = _latency()
            painting.stop()
            say.item("event wait, backdrop running", f"{busy:8.2f} ms")
            over = "  <-- a frame is 16.7 ms" if busy > 16.7 else ""
            say.item("added by the backdrop", f"{busy - idle:8.2f} ms{over}")
        except Exception as exc:                             # noqa: BLE001
            say.failed("event wait under backdrop", exc)

        # And while a screen is being built, which is the worst moment: the
        # user has just clicked a module and everything is competing.
        try:
            from spacr.qt.screens.app_screen import AppScreen
            start = _clock()
            screen = AppScreen("mask")
            app.processEvents()
            say.timed("build mask with a window up", _clock() - start)
            say.item("widgets alive after", f"{len(app.allWidgets()):,}")
            del screen
        except Exception as exc:                             # noqa: BLE001
            say.failed("build under load", exc)

        say("")
        say("  Read these together: each animation above may be well inside")
        say("  a frame on its own. What decides whether the application feels")
        say("  smooth is the wait a queued event sees with all of them going.")
        window.close()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv
    say = Report()

    say("=" * 70)
    say("spaCR hardware report")
    say("=" * 70)
    say(f"generated {time.strftime('%Y-%m-%d %H:%M:%S')}"
        + ("  (--quick)" if quick else ""))

    machine(say)
    imports(say)
    numerics(say, quick)
    display(say)
    preferences_in_effect(say)
    theming(say)
    the_real_launch(say)
    the_splash(say)
    startup(say)
    screens(say, quick)
    backdrop(say)
    cursor_rim(say)
    concurrency(say)

    say("")
    say("=" * 70)
    try:
        if "--out" in argv:
            path = argv[argv.index("--out") + 1]
        else:
            folder = os.path.join(os.path.expanduser("~"), ".spacr", "reports")
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(
                folder, f"hardware-{time.strftime('%Y%m%d-%H%M%S')}.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(say.lines) + "\n")
        say(f"Saved to {path}")
        say("Attach that file to your GitHub issue, or paste the text above.")
    except Exception as exc:                                 # noqa: BLE001
        say(f"Could not save a copy: {exc}. Copy the report text above instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
