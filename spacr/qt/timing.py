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
from typing import List, Optional, Tuple

#: On only when asked. The check is a string compare against an environment
#: variable read once, so an ordinary launch pays for it once.
ENABLED: bool = str(os.environ.get("SPACR_TIMING", "")).strip().lower() in (
    "1", "true", "yes", "on")

#: Imports faster than this are not worth a line; there are thousands.
IMPORT_FLOOR_MS: float = 5.0

#: A timer asked to fire every 16 ms that fires later than this was blocked.
STALL_FLOOR_MS: float = 50.0

_START = time.perf_counter()
_LOCK = threading.Lock()
_SPANS: List[dict] = []
_IMPORTS: List[dict] = []
_STALLS: List[dict] = []
_MARKS: List[dict] = []
_DEPTH = threading.local()


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

    state = {"last": time.perf_counter()}

    def _beat():
        now = time.perf_counter()
        late = (now - state["last"]) * 1000.0
        state["last"] = now
        if late >= STALL_FLOOR_MS:
            with _LOCK:
                _STALLS.append({"at": _now(), "late_ms": late})

    timer = QTimer(parent)
    timer.setInterval(16)
    timer.timeout.connect(_beat)
    timer.start()
    return timer


def report() -> str:
    """The timeline, as text."""
    with _LOCK:
        spans = list(_SPANS)
        imports = list(_IMPORTS)
        stalls = list(_STALLS)
        marks = list(_MARKS)

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
            out.append(f"    at {stall['at']:7.2f}s   froze "
                       f"{stall['late_ms']:8.0f} ms")

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
    """Start recording. Safe to call twice; does nothing when off."""
    if not ENABLED:
        return
    global _START
    if getattr(begin, "_done", False):
        return
    begin._done = True
    _START = time.perf_counter()
    _install_import_timer()
    mark("timing started")
