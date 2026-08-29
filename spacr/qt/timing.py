"""What the application actually did, and when, with nothing inferred.

Switched on with ``SPACR_TIMING=1``. Off, every function here is a few
attribute lookups and nothing is imported, allocated or written.

WHY THIS EXISTS. Every measurement of spaCR's start-up so far was taken in
a script that imitated the application -- built a MainWindow, called a
method, timed it -- and each one answered a question next to the one being
asked. A script cannot see the preloader thread competing with a click, a
stylesheet reapplied four times, or a lazy import that fires the first time
a user opens a panel. This records the real process.

FOUR THINGS ARE RECORDED, all with real wall-clock times:

* SPANS -- named, nested regions of work. ``with span("build mask"):``
* IMPORTS -- every module import over a threshold, with the importing frame,
  through a `sys.meta_path` finder rather than by patching ``__import__``.
* GUI STALLS -- a timer that should fire every 16 ms, recording every time
  it is late. This is the only honest measure of "the app froze": it is the
  event loop reporting on itself.
* MARKS -- instants worth a line, like "window shown".

The report is a TIMELINE, not a profile. A profile says which function used
the CPU; this says what the user was waiting for and for how long, which is
a different question and the one being asked.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Callable, List, Optional

#: On only when asked. The check is a string compare against an environment
#: variable read once, so an ordinary launch pays for it once.
ENABLED: bool = str(os.environ.get("SPACR_TIMING", "")).strip().lower() in (
    "1", "true", "yes", "on")

#: Import attribution is diagnostic profiling, not needed for every
#: acceptance run.  The loader wrapper necessarily adds work to every import,
#: so the benchmark driver disables it while retaining the process clock,
#: readiness and event-loop watchdog.  An ordinary ``SPACR_TIMING=1`` report
#: remains exhaustive by default.
IMPORT_TIMING_ENABLED: bool = str(
    os.environ.get("SPACR_TIMING_IMPORTS", "1")
).strip().lower() not in ("0", "false", "no", "off")

#: Imports faster than this are not worth a line; there are thousands.
IMPORT_FLOOR_MS: float = 5.0

#: A timer asked to fire every 16 ms that fires later than this was blocked.
STALL_FLOOR_MS: float = 50.0

#: User-visible acceptance budgets from instructions 284 and 305.  The
#: watchdog records smaller gaps so a trace remains diagnostic; only a gap at
#: or above this value violates the release contract.
HOME_BUDGET_S: float = 5.0
MODULE_BUDGET_S: float = 10.0
STALL_BUDGET_MS: float = 500.0

_START = time.perf_counter()
_LOCK = threading.Lock()
_SPANS: List[dict] = []
_IMPORTS: List[dict] = []
_STALLS: List[dict] = []
_MARKS: List[dict] = []
_READINESS: List[dict] = []
_READY_CALLBACKS: List[Callable[[dict], None]] = []
_ACTIVE_PROBES: List[object] = []
_DEPTH = threading.local()
_EVENT_LOOP_STARTED_AT: Optional[float] = None
_LAST_GUI_BEAT_AT: Optional[float] = None
_GUI_WATCHDOG_ACTIVE = False
_IMPORT_TIMER_INSTALLED = False


def _now() -> float:
    return time.perf_counter() - _START


def _depth() -> int:
    return getattr(_DEPTH, "value", 0)


def mark(name: str, detail: str = "") -> None:
    """Record an instant. Cheap enough to leave in hot paths."""
    if not ENABLED:
        return
    with _LOCK:
        _MARKS.append({"at": _now(), "name": str(name),
                       "detail": str(detail),
                       "thread": threading.current_thread().name})


def interval_started(name: str, detail: str = "") -> Optional[float]:
    """Return an absolute start time for a user-visible interaction.

    ``None`` while timing is disabled keeps the ordinary navigation path to
    one branch and no allocation.  The absolute value is deliberately opaque
    to callers; :func:`watch_interactive` turns it into a report duration.
    """
    if not ENABLED:
        return None
    started = time.perf_counter()
    mark(f"{name} requested", detail)
    return started


def process_started_at() -> Optional[float]:
    """The absolute clock origin used by this timing session, when enabled."""
    return _START if ENABLED else None


def elapsed() -> float:
    """Seconds on the instrumentation clock without building a snapshot."""
    return _now()


def stalls_between(started_at: float, ended_at: float,
                   stalls: Optional[List[dict]] = None) -> List[dict]:
    """Return watchdog gaps clipped to their overlap with one interval.

    A watchdog beat can begin before a click and end after it. Charging that
    whole gap to the click can report a multi-second freeze for an interaction
    that lasted only a few hundred milliseconds. Preserve the raw interval in
    ``late_ms`` and add the honest in-window portion as ``overlap_ms``.
    """
    start = float(started_at)
    end = float(ended_at)
    if end <= start:
        return []
    if stalls is None:
        with _LOCK:
            source = [dict(row) for row in _STALLS]
    else:
        source = [dict(row) for row in stalls]

    overlapping = []
    for row in source:
        gap_end = float(row.get("at", -1.0))
        raw_ms = max(0.0, float(row.get("late_ms", 0.0)))
        gap_start = float(row.get(
            "started_at", gap_end - raw_ms / 1000.0))
        overlap_ms = max(
            0.0, min(end, gap_end) - max(start, gap_start)) * 1000.0
        if overlap_ms <= 0.0:
            continue
        row["started_at"] = gap_start
        row["overlap_ms"] = min(raw_ms, overlap_ms)
        overlapping.append(row)
    return overlapping


def last_gui_beat_at() -> Optional[float]:
    """Elapsed timestamp of the latest watchdog beat, or ``None`` if absent."""
    if not _GUI_WATCHDOG_ACTIVE or _LAST_GUI_BEAT_AT is None:
        return None
    return _LAST_GUI_BEAT_AT - _START


@contextmanager
def span(name: str, detail: str = ""):
    """Time a named region, nested under whatever encloses it.

    Records even when the body raises: a span that only appears on success
    hides exactly the slow failures worth seeing.
    """
    if not ENABLED:
        yield
        return
    _DEPTH.value = _depth() + 1
    started = _now()
    failed = ""
    try:
        yield
    except BaseException as error:                           # noqa: BLE001
        failed = type(error).__name__
        raise
    finally:
        _DEPTH.value = _depth() - 1
        with _LOCK:
            _SPANS.append({
                "at": started, "took": _now() - started, "name": str(name),
                "detail": str(detail), "depth": _depth(), "failed": failed,
                "thread": threading.current_thread().name,
            })


class _ImportTimer:
    """A `sys.meta_path` finder that times every import it sees.

    A finder rather than a patched ``__import__``: the finder sees the
    module being LOADED, which is the part that costs, and it does not
    change import semantics for anything else in the process.
    """

    def __init__(self) -> None:
        self._depth = 0

    def find_module(self, fullname, path=None):              # noqa: D102
        return None

    def find_spec(self, fullname, path=None, target=None):
        if fullname in sys.modules:
            return None
        started = time.perf_counter()
        caller = ""
        try:
            frame = sys._getframe(1)
            for _ in range(12):
                if frame is None:
                    break
                name = frame.f_code.co_filename
                if "/spacr/" in name and "timing.py" not in name:
                    caller = (f"{name.split('/spacr/')[-1]}"
                              f":{frame.f_lineno}")
                    break
                frame = frame.f_back
        except Exception:                                    # noqa: BLE001
            pass
        # Let the real finders answer; we only time how long that takes and
        # how long the module then takes to execute, which the next call
        # into this finder for a submodule will nest under.
        self._pending = (fullname, started, caller)
        return None

    def note(self) -> None:
        pass


def _install_import_timer() -> None:
    """Record every import over the floor, with what asked for it."""
    global _IMPORT_TIMER_INSTALLED
    if _IMPORT_TIMER_INSTALLED:
        return
    _IMPORT_TIMER_INSTALLED = True
    import importlib.abc
    import importlib.machinery

    real_exec = importlib.machinery.SourceFileLoader.exec_module
    real_ext = None
    try:
        real_ext = importlib.machinery.ExtensionFileLoader.exec_module
    except Exception:                                        # noqa: BLE001
        pass

    def _timed(original):
        def exec_module(self, module):
            name = getattr(module, "__name__", "?")
            started = time.perf_counter()
            try:
                return original(self, module)
            finally:
                took = (time.perf_counter() - started) * 1000.0
                if took >= IMPORT_FLOOR_MS:
                    caller = ""
                    try:
                        frame = sys._getframe(2)
                        for _ in range(15):
                            if frame is None:
                                break
                            path = frame.f_code.co_filename
                            if ("/spacr/" in path
                                    and "timing.py" not in path):
                                caller = (path.split("/spacr/")[-1]
                                          + f":{frame.f_lineno}")
                                break
                            frame = frame.f_back
                    except Exception:                        # noqa: BLE001
                        pass
                    with _LOCK:
                        _IMPORTS.append({
                            "at": _now(), "took": took / 1000.0,
                            "name": name, "by": caller,
                            "thread": threading.current_thread().name})
        return exec_module

    importlib.machinery.SourceFileLoader.exec_module = _timed(real_exec)
    if real_ext is not None:
        importlib.machinery.ExtensionFileLoader.exec_module = _timed(real_ext)


def watch_the_gui_thread(parent=None):
    """Start the stall watchdog. Returns the timer, or None when off.

    A QTimer asked for 16 ms records how late it actually was. THIS IS THE
    ONLY HONEST FREEZE MEASUREMENT: it is the event loop reporting on
    itself, from inside the real application, so it cannot miss a stall the
    way an outside script can.
    """
    if not ENABLED:
        return None
    from PySide6.QtCore import QTimer

    global _GUI_WATCHDOG_ACTIVE, _LAST_GUI_BEAT_AT
    _LAST_GUI_BEAT_AT = time.perf_counter()
    _GUI_WATCHDOG_ACTIVE = True
    state = {"last": _LAST_GUI_BEAT_AT}

    def _beat():
        global _LAST_GUI_BEAT_AT
        now = time.perf_counter()
        previous = state["last"]
        late = (now - previous) * 1000.0
        state["last"] = now
        _LAST_GUI_BEAT_AT = now
        if late >= STALL_FLOOR_MS:
            with _LOCK:
                _STALLS.append({
                    "at": now - _START,
                    "started_at": previous - _START,
                    "late_ms": late,
                    "source": "event-loop watchdog",
                    "thread": threading.current_thread().name,
                })

    timer = QTimer(parent)
    timer.setInterval(16)
    timer.timeout.connect(_beat)
    timer.start()
    return timer


def event_loop_started() -> None:
    """Record the first callback actually delivered by the Qt event loop.

    ``launch`` schedules this with a zero-delay timer immediately before
    ``QApplication.exec``.  Unlike a mark placed before ``exec()``, reaching
    this function proves that the loop has begun dispatching events.
    """
    global _EVENT_LOOP_STARTED_AT
    if not ENABLED or _EVENT_LOOP_STARTED_AT is not None:
        return
    _EVENT_LOOP_STARTED_AT = time.perf_counter()
    mark("event loop began")
    for probe in list(_ACTIVE_PROBES):
        try:
            probe.event_loop_started()
        except RuntimeError:
            try:
                _ACTIVE_PROBES.remove(probe)
            except ValueError:
                pass


def subscribe_readiness(callback: Callable[[dict], None]) -> None:
    """Call ``callback`` after each post-paint interactive-ready record."""
    if callback not in _READY_CALLBACKS:
        _READY_CALLBACKS.append(callback)


def unsubscribe_readiness(callback: Callable[[dict], None]) -> None:
    """Remove a callback installed by :func:`subscribe_readiness`."""
    try:
        _READY_CALLBACKS.remove(callback)
    except ValueError:
        pass


def cancel_interactive(*, name: str = "", detail: str = "") -> int:
    """Retire unfinished readiness probes matching ``name`` / ``detail``."""
    retired = 0
    for probe in list(_ACTIVE_PROBES):
        if name and getattr(probe, "report_name", "") != name:
            continue
        if detail and getattr(probe, "report_detail", "") != detail:
            continue
        try:
            probe._retire()
        except RuntimeError:
            try:
                _ACTIVE_PROBES.remove(probe)
            except ValueError:
                pass
        retired += 1
    return retired


def watch_interactive(
    widget,
    name: str,
    detail: str = "",
    *,
    started_at: Optional[float] = None,
    budget_s: Optional[float] = None,
):
    """Observe when ``widget`` is genuinely painted and operable.

    Readiness requires all of the following, observed rather than inferred:

    * Qt has delivered a callback after the application event loop began;
    * the screen's visible widget tree has delivered a paint event;
    * at least one enabled, visible, non-zero-sized input control has painted;
    * one further event-loop turn has run after those paint events.

    The observer is installed after construction but before the new page can
    paint.  It removes itself at the first valid state and is parented to the
    observed widget, so neither a report nor a failed screen keeps a window
    alive.  PySide6 is imported only while timing is explicitly enabled.
    """
    if not ENABLED or widget is None:
        return None

    cancel_interactive(name=str(name), detail=str(detail))

    from PySide6.QtCore import QEvent, QObject, QTimer
    from PySide6.QtWidgets import (
        QAbstractButton,
        QAbstractItemView,
        QAbstractSlider,
        QAbstractSpinBox,
        QComboBox,
        QLineEdit,
        QTabBar,
        QWidget,
    )

    control_types = (
        QAbstractButton, QAbstractItemView, QAbstractSlider,
        QAbstractSpinBox, QComboBox, QLineEdit, QTabBar,
    )
    controls = [
        child for child in widget.findChildren(QWidget)
        if isinstance(child, control_types)
    ]
    if isinstance(widget, control_types):
        controls.insert(0, widget)

    class _InteractivePaintProbe(QObject):
        def __init__(self) -> None:
            super().__init__(widget)
            self.root = widget
            self.report_name = str(name)
            self.report_detail = str(detail)
            self.controls = tuple(controls)
            self.control_ids = {id(control) for control in controls}
            self.root_painted = False
            self.painted_controls: set[int] = set()
            self.done = False
            self._settle_queued = False

        def eventFilter(self, watched, event):  # noqa: N802 - Qt naming
            if event.type() != QEvent.Type.Paint:
                return False
            if watched is self.root:
                self.root_painted = True
            if id(watched) in self.control_ids:
                self.painted_controls.add(id(watched))
            self._queue_settle()
            return False

        def event_loop_started(self) -> None:
            # A paint may have been delivered by show() before exec().  It is
            # evidence about the widget, but not evidence for this contract:
            # readiness begins only after the application event loop has
            # actually dispatched a callback.  Discard every pre-loop paint
            # before forcing another one, or an already-painted control plus
            # the settle timer below could report a false ready state without
            # a post-exec paint ever being observed.
            self.root_painted = False
            self.painted_controls.clear()
            try:
                self.root.update()
                for control in self.controls:
                    if control.isVisible():
                        control.update()
            except RuntimeError:
                self._retire()
                return
            self._queue_settle()

        def _queue_settle(self) -> None:
            if self.done or self._settle_queued:
                return
            self._settle_queued = True
            QTimer.singleShot(0, self._settle)

        @staticmethod
        def _usable(control) -> bool:
            try:
                size = control.size()
                return (
                    control.isEnabled()
                    and control.isVisible()
                    and size.width() > 0
                    and size.height() > 0
                )
            except RuntimeError:
                return False

        def _settle(self) -> None:
            self._settle_queued = False
            if self.done or _EVENT_LOOP_STARTED_AT is None:
                return
            try:
                root_usable = self.root.isVisible() and self.root.isEnabled()
            except RuntimeError:
                self._retire()
                return
            painted_usable = [
                control for control in self.controls
                if id(control) in self.painted_controls
                and self._usable(control)
            ]
            # A transparent container legitimately receives no paint event of
            # its own: Home is exactly such a widget under an ambient theme.
            # A descendant control's completed paint is stronger evidence
            # than forcing the root to repaint for the benchmark, and it is
            # the state the contract asks for -- a control the user can see
            # and operate.  Keep root_painted as factual diagnostic evidence,
            # but do not invent a root paint where Qt optimised one away.
            if not root_usable or not painted_usable:
                return

            now = time.perf_counter()
            origin = _START if started_at is None else started_at
            effective_budget = budget_s
            duration = max(0.0, now - origin)
            entry = {
                "at": now - _START,
                "started_at": origin - _START,
                "duration_s": duration,
                "name": self.report_name,
                "detail": self.report_detail,
                "budget_s": effective_budget,
                "within_budget": (
                    None if effective_budget is None
                    else duration <= effective_budget
                ),
                "event_loop_started_at": (
                    _EVENT_LOOP_STARTED_AT - _START
                    if _EVENT_LOOP_STARTED_AT is not None else None
                ),
                "root_painted": self.root_painted,
                "screen_tree_painted": True,
                "painted_usable_controls": len(painted_usable),
                "usable_controls": len([
                    control for control in self.controls
                    if self._usable(control)
                ]),
                "controls": [
                    str(control.objectName() or type(control).__name__)
                    for control in painted_usable[:8]
                ],
                "thread": threading.current_thread().name,
            }
            with _LOCK:
                _READINESS.append(entry)
            mark(name, detail)
            self._retire()
            for callback in list(_READY_CALLBACKS):
                try:
                    callback(dict(entry))
                except Exception:                            # noqa: BLE001
                    # Instrumentation may never make navigation fail.  The
                    # benchmark controller records its own errors separately.
                    continue

        def _retire(self) -> None:
            if self.done:
                return
            self.done = True
            for watched in (self.root, *self.controls):
                try:
                    watched.removeEventFilter(self)
                except RuntimeError:
                    pass
            try:
                _ACTIVE_PROBES.remove(self)
            except ValueError:
                pass
            self.deleteLater()

    probe = _InteractivePaintProbe()
    for watched in (widget, *controls):
        try:
            watched.installEventFilter(probe)
        except RuntimeError:
            continue
    _ACTIVE_PROBES.append(probe)
    if _EVENT_LOOP_STARTED_AT is not None:
        QTimer.singleShot(0, probe.event_loop_started)
    return probe


def report() -> str:
    """The timeline, as text."""
    with _LOCK:
        spans = list(_SPANS)
        imports = list(_IMPORTS)
        stalls = list(_STALLS)
        marks = list(_MARKS)
        readiness = list(_READINESS)

    out: List[str] = []
    out.append("=" * 78)
    out.append("spaCR timing report -- real process, not a proxy")
    out.append(f"total wall clock: {_now():.2f}s")
    out.append("=" * 78)

    out.append("")
    out.append("GUI THREAD STALLS  (a 16 ms timer reporting how late it was)")
    if not stalls:
        out.append("  none over "
                   f"{STALL_FLOOR_MS:.0f} ms -- the interface stayed answerable")
    else:
        worst = sorted(stalls, key=lambda s: -s["late_ms"])
        out.append(f"  {len(stalls)} stalls; "
                   f"total frozen {sum(s['late_ms'] for s in stalls)/1000:.2f}s")
        for stall in worst[:20]:
            source = f"  [{stall.get('source', 'watchdog')}]"
            out.append(f"    at {stall['at']:7.2f}s   froze "
                       f"{stall['late_ms']:8.0f} ms{source}")

    out.append("")
    out.append("INTERACTIVE READINESS  (after event-loop + screen/control paint)")
    if not readiness:
        out.append("  none observed")
    for entry in readiness:
        budget = entry.get("budget_s")
        verdict = ""
        if budget is not None:
            verdict = ("  OK" if entry.get("within_budget") else
                       f"  OVER {budget:.1f}s BUDGET")
        out.append(
            f"  {entry['at']:7.2f}s  {entry['duration_s']*1000:8.1f} ms  "
            f"{entry['name']}  [{entry['detail']}]  "
            f"{entry['painted_usable_controls']} painted control(s){verdict}")

    out.append("")
    out.append("SPANS  (nested, in the order they finished)")
    for entry in sorted(spans, key=lambda s: s["at"]):
        pad = "  " * entry["depth"]
        note = f"  [{entry['detail']}]" if entry["detail"] else ""
        bad = f"  RAISED {entry['failed']}" if entry["failed"] else ""
        thread = ("" if entry["thread"] == "MainThread"
                  else f"  <{entry['thread']}>")
        out.append(f"  {entry['at']:7.2f}s  {entry['took']*1000:8.1f} ms  "
                   f"{pad}{entry['name']}{note}{thread}{bad}")

    out.append("")
    out.append(f"IMPORTS over {IMPORT_FLOOR_MS:.0f} ms  "
               f"({len(imports)} of them, "
               f"{sum(i['took'] for i in imports):.2f}s total)")
    for entry in sorted(imports, key=lambda i: -i["took"])[:40]:
        thread = ("" if entry["thread"] == "MainThread"
                  else f"  <{entry['thread']}>")
        by = f"  asked by {entry['by']}" if entry["by"] else ""
        out.append(f"    {entry['took']*1000:8.1f} ms  at {entry['at']:6.2f}s  "
                   f"{entry['name']}{by}{thread}")

    out.append("")
    out.append("MARKS")
    for entry in sorted(marks, key=lambda m: m["at"]):
        detail = f"  {entry['detail']}" if entry["detail"] else ""
        out.append(f"  {entry['at']:7.2f}s  {entry['name']}{detail}")
    out.append("")
    return "\n".join(out)


def _peak_rss_mb() -> Optional[float]:
    """Peak process resident memory without importing a monitoring stack."""
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # macOS reports bytes; Linux and the supported BSD runners report KiB.
        return value / (1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0)
    except Exception:                                       # noqa: BLE001
        try:
            import psutil

            info = psutil.Process().memory_info()
            # Windows exposes the process peak as peak_wset.  On a platform
            # without that field, current RSS is explicitly the best
            # available fallback rather than a fabricated peak.
            value = getattr(info, "peak_wset", None)
            if value is None:
                value = info.rss
            return float(value) / (1024.0 * 1024.0)
        except Exception:                                   # noqa: BLE001
            return None


def _gpu_memory_mb() -> dict:
    """Report an already-initialised Torch CUDA allocator without loading it."""
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    if cuda is None:
        return {"allocated_mb": None, "peak_allocated_mb": None}

    try:
        if not cuda.is_initialized():
            return {"allocated_mb": 0.0, "peak_allocated_mb": 0.0}
        scale = 1024.0 * 1024.0
        return {
            "allocated_mb": float(cuda.memory_allocated()) / scale,
            "peak_allocated_mb": float(cuda.max_memory_allocated()) / scale,
        }
    except Exception:                                       # noqa: BLE001
        return {"allocated_mb": None, "peak_allocated_mb": None}


def _hardware_profile() -> dict:
    """Hardware/display facts already available without loading a backend."""
    total_memory_mb = None
    try:
        total_memory_mb = (
            os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            / (1024.0 * 1024.0)
        )
    except (AttributeError, OSError, TypeError, ValueError):
        try:
            import psutil

            total_memory_mb = psutil.virtual_memory().total / (1024.0 * 1024.0)
        except Exception:                                   # noqa: BLE001
            pass

    displays = []
    qt_widgets = sys.modules.get("PySide6.QtWidgets")
    application = getattr(qt_widgets, "QApplication", None)
    app = application.instance() if application is not None else None
    if app is not None:
        try:
            for screen in app.screens():
                geometry = screen.geometry()
                displays.append({
                    "name": str(screen.name()),
                    "logical_width": geometry.width(),
                    "logical_height": geometry.height(),
                    "device_pixel_ratio": float(screen.devicePixelRatio()),
                    "refresh_hz": float(screen.refreshRate()),
                })
        except Exception:                                   # noqa: BLE001
            displays = []

    performance_level = None
    preferences = sys.modules.get("spacr.qt.preferences")
    getter = getattr(preferences, "get_performance_level", None)
    if callable(getter):
        try:
            performance_level = str(getter())
        except Exception:                                   # noqa: BLE001
            pass
    return {
        "logical_cpu_count": os.cpu_count(),
        "total_memory_mb": total_memory_mb,
        "performance_level": performance_level,
        "qt_platform": str(app.platformName()) if app is not None else None,
        "displays": displays,
    }


def snapshot() -> dict:
    """Return the complete timing state as a JSON-serialisable artifact."""
    # Keep disabled timing stdlib-light: platform performs several imports,
    # and snapshots exist only for an explicitly enabled diagnostic run.
    import platform

    with _LOCK:
        spans = [dict(value) for value in _SPANS]
        imports = [dict(value) for value in _IMPORTS]
        stalls = [dict(value) for value in _STALLS]
        marks = [dict(value) for value in _MARKS]
        readiness = [dict(value) for value in _READINESS]

    worst_stall = max((row["late_ms"] for row in stalls), default=0.0)
    qt_version = None
    qt_core = sys.modules.get("PySide6.QtCore")
    if qt_core is not None:
        try:
            qt_version = str(qt_core.qVersion())
        except Exception:                                   # noqa: BLE001
            pass
    package = sys.modules.get("spacr")
    qt_package = sys.modules.get("spacr.qt")
    return {
        "schema_version": 1,
        "elapsed_s": _now(),
        "budgets": {
            "home_ready_s": HOME_BUDGET_S,
            "module_ready_s": MODULE_BUDGET_S,
            "max_event_loop_stall_ms": STALL_BUDGET_MS,
            "watchdog_record_floor_ms": STALL_FLOOR_MS,
        },
        "import_timing_enabled": IMPORT_TIMING_ENABLED,
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "qt": qt_version,
            "executable": sys.executable,
            "pid": os.getpid(),
            "spacr_file": getattr(package, "__file__", None),
            "qt_package_file": getattr(qt_package, "__file__", None),
            "spacr_version": getattr(package, "__version__", None),
            "hardware": _hardware_profile(),
        },
        "resources": {
            "peak_rss_mb": _peak_rss_mb(),
            "gpu": _gpu_memory_mb(),
        },
        "event_loop_started_at": (
            None if _EVENT_LOOP_STARTED_AT is None
            else _EVENT_LOOP_STARTED_AT - _START
        ),
        "worst_event_loop_stall_ms": worst_stall,
        "stall_budget_met": worst_stall < STALL_BUDGET_MS,
        "spans": spans,
        "imports": imports,
        "stalls": stalls,
        "marks": marks,
        "readiness": readiness,
    }


def write_json(path: str) -> str:
    """Write :func:`snapshot` to ``path`` and return it, or ``""`` on error."""
    if not ENABLED or not path:
        return ""
    try:
        import json

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(snapshot(), handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (OSError, TypeError, ValueError):
        return ""
    return path


def write_report(path: str = "") -> str:
    """Write the timeline. Returns the path, or "" when timing is off."""
    if not ENABLED:
        return ""
    target = path or os.environ.get("SPACR_TIMING_LOG", "") or os.path.join(
        os.getcwd(), "spacr-timing.log")
    try:
        with open(target, "w") as handle:
            handle.write(report())
    except OSError:
        return ""
    return target


def begin() -> None:
    """Start recording before the public launch imports or registers Qt.

    A benchmark wrapper may provide ``SPACR_TIMING_PROCESS_START`` as a wall
    clock captured immediately before spawning this interpreter.  The child
    translates its age onto :func:`time.perf_counter`; unlike sharing a raw
    monotonic value, that also works on Python 3.9 Windows.  An ordinary
    ``SPACR_TIMING=1 spacr`` starts at this module's own import, still before
    the expensive Qt application path.
    """
    if not ENABLED:
        return
    global _START
    if getattr(begin, "_done", False):
        return
    begin._done = True
    source = "timing module import"
    raw_start = os.environ.get("SPACR_TIMING_PROCESS_START", "")
    if raw_start:
        try:
            candidate = float(raw_start)
            age = time.time() - candidate
            if 0.0 <= age < 3600.0:
                _START = time.perf_counter() - age
                source = "benchmark process spawn"
        except (TypeError, ValueError):
            pass
    if IMPORT_TIMING_ENABLED:
        _install_import_timer()
    mark("timing started", source)
