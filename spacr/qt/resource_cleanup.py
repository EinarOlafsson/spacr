"""Free what spaCR owns, and nothing else.

Four buttons in Preferences — **clear RAM**, **clear VRAM**, **clear CPU**,
**check disk space** — and the two performance modes that press them for you.
This module is what they call.

The rule, and it is a refusal rather than a preference
------------------------------------------------------

"Free as many resources as possible" must never reach anything spaCR does
not own. This machine runs other people's work: a segmentation that starts
four seconds sooner is not worth somebody's eight-hour training run, and a
tool that decides otherwise on the user's behalf is not a tool anybody can
leave running.

So, concretely, and asserted in ``tests/qt/test_resource_cleanup.py`` by
reading this source file:

* **No process is ever killed.** Not by name, not by "python processes using
  a lot of memory", not spaCR's own children. There is no ``os.kill``, no
  ``signal``, no ``subprocess``, no ``Process.terminate()``, no
  ``QThread.terminate()`` anywhere in this module or reachable from it. The
  Qt layer removed ``QThread.terminate()`` outright; a stubborn thread is
  *parked* by :func:`spacr.qt.bridge.drain_thread`, and that is the strongest
  thing any of this may do.
* **Nothing here needs root**, and nothing here touches the operating
  system's own memory. No ``drop_caches``, no ``sysctl``, no ``swapoff``.
  The page cache belongs to the kernel and dropping it would slow down the
  whole machine, spaCR included — it is not a free win, it is a transfer
  from everybody to nobody.
* **A run in flight is never disturbed.** No queued job is dropped, no
  worker is cancelled, no thread pool is emptied of work that has not
  started yet. :func:`spacr.qt.bridge.registry` is *read* here and never
  cancelled.

What each button can honestly do
--------------------------------

``clear RAM``
    ``gc.collect()``, plus dropping spaCR's own caches: the merged-field LRU
    (:func:`spacr.crops.clear_field_cache` — by far the largest, whole image
    stacks), the file-format and DB-format caches, the zoomed-animation
    cache, the icon and preview ``lru_cache``s, the filter-kind cache, every
    live :class:`~spacr.qt.crop_thumbs.CropThumbnails` thumbnail LRU, and
    Qt's own ``QPixmapCache``. It cannot return memory the allocator has
    decided to keep, and it says so when the measured RSS does not move.

``clear VRAM``
    ``torch.cuda.empty_cache()``, which hands the CUDA driver back the
    blocks torch reserved and is no longer using, plus releasing any model
    reference spaCR itself is holding (:data:`MODEL_RELEASERS`). **It cannot
    reclaim another process's VRAM** — no process can — and it cannot free a
    tensor a running spaCR job is still using, which is the point rather
    than a limitation.

``clear CPU``
    Releases parked worker threads that have since exited
    (:func:`spacr.qt.bridge.prune_parked_threads`), lets Qt's global pool
    retire its idle threads, and lowers spaCR's own library thread counts
    (torch, OpenCV) to a floor. It retires *idle* capacity only: work that
    is queued or running is left exactly where it is.

``check disk space``
    Read-only. ``shutil.disk_usage`` over the filesystems the current
    project actually touches, deduplicated by device so one line is one
    drive. It frees nothing and never claims to.

Reporting
---------

Every number is measured before and after by the same function, and
:class:`Reclaim` carries both endpoints so a caller can show the subtraction
rather than an estimate. A cleanup that freed nothing reports that it freed
nothing; there is no reassuring dialog. See :meth:`Reclaim.summary`.
"""
from __future__ import annotations

import gc
import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

LOG = logging.getLogger("spacr.qt.resource_cleanup")

__all__ = [
    "Reclaim", "DiskEntry", "DiskReport",
    "clear_ram", "clear_vram", "clear_cpu", "disk_report",
    "confirmation_title", "confirmation_text", "ACTIONS",
    "MODEL_RELEASERS", "process_rss", "cuda_reserved",
    "run_launch_cleanup", "run_pre_run_cleanup", "install_run_hook",
    "register",
]

#: The four buttons, in the order Preferences shows them.
ACTIONS: Tuple[str, ...] = ("ram", "vram", "cpu", "disk")

#: Callables that release a model reference spaCR itself is holding, each
#: returning how many it released. Empty, and the emptiness is a finding:
#: spaCR builds its Cellpose and torch models *inside* the run function and
#: drops them when the run ends, so between runs there is no long-lived
#: model to release and ``empty_cache()`` is the whole of the reclaim. The
#: hook exists so that a screen which starts caching a warm model has one
#: obvious place to say so, rather than this module growing a list of
#: attribute names to go rummaging for.
MODEL_RELEASERS: List[Callable[[], int]] = []

#: Modules whose ``functools.lru_cache``-decorated functions are spaCR's own
#: derived pixmaps and parsed assets — safe to drop, rebuilt on demand.
_LRU_CACHE_MODULES: Tuple[str, ...] = (
    "spacr.qt.iconset",
    "spacr.qt.widgets.preview_controls",
    "spacr.qt.widgets.animation_zoom",
)

#: ``module``, ``attribute`` pairs naming a plain dict spaCR uses as a cache.
_DICT_CACHES: Tuple[Tuple[str, str], ...] = (
    ("spacr.crops", "_FIELD_CACHE"),
    ("spacr.crops", "_FORMAT_CACHE"),
    ("spacr.crops", "_DB_FORMAT_CACHE"),
    ("spacr.qt.widgets.data_filter_panel", "_KINDS_CACHE"),
)


# ---------------------------------------------------------------------------
# What a cleanup reports
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Reclaim:
    """The measured result of one cleanup. Not the intent — the outcome.

    :param action: one of :data:`ACTIONS`.
    :param before: the measurement taken before anything ran, in bytes.
    :param after: the same measurement, taken after.
    :param details: what was actually done, one short phrase each.
    :param note: the honest caveat, if there is one. A cleanup that could
        do nothing in the current state says so here.
    :param measured: ``False`` when there was no measurement to take at all
        (no CUDA device, no way to read RSS), which is different from a
        measurement that came back zero.
    """

    action: str
    before: int = 0
    after: int = 0
    details: Tuple[str, ...] = ()
    note: str = ""
    measured: bool = True

    @property
    def freed(self) -> int:
        """Bytes actually returned, measured. Never negative — memory that
        *grew* across the call is reported as zero freed and named in
        :meth:`summary`, because "freed -4 MB" is not a thing."""
        return max(0, int(self.before) - int(self.after))

    @property
    def grew(self) -> int:
        """Bytes the measurement went UP by, if it did."""
        return max(0, int(self.after) - int(self.before))

    def summary(self) -> str:
        """One line a dialog can show, and it is allowed to be bad news."""
        label = {"ram": "RAM", "vram": "VRAM", "cpu": "CPU"}.get(
            self.action, self.action)
        if not self.measured:
            return f"{label}: nothing to measure — {self.note}"
        if self.action == "cpu":
            plural = "" if self.after == 1 else "s"
            body = (f"{self.before} → {self.after} threads"
                    if self.before != self.after
                    else f"{self.after} thread{plural}, unchanged")
        elif self.freed:
            body = f"freed {human_bytes(self.freed)}"
        elif self.grew:
            body = (f"freed nothing — {human_bytes(self.grew)} more is in "
                    "use than before")
        else:
            body = "freed nothing measurable"
        if self.note:
            return f"{label}: {body}. {self.note}"
        return f"{label}: {body}."


@dataclass(frozen=True)
class DiskEntry:
    """One filesystem, and what it is holding."""

    path: str
    total: int
    used: int
    free: int

    @property
    def percent_used(self) -> float:
        return (100.0 * self.used / self.total) if self.total else 0.0

    def summary(self) -> str:
        return (f"{self.path}: {human_bytes(self.free)} free of "
                f"{human_bytes(self.total)} ({self.percent_used:.0f}% used)")


@dataclass(frozen=True)
class DiskReport:
    """Every drive the current project touches. Read-only, always."""

    entries: Tuple[DiskEntry, ...] = ()
    note: str = ""

    def summary(self) -> str:
        if not self.entries:
            return self.note or "No project folder is known yet."
        lines = [entry.summary() for entry in self.entries]
        if self.note:
            lines.append(self.note)
        return "\n".join(lines)

    @property
    def tightest(self) -> Optional[DiskEntry]:
        """The drive with the least room, which is the one worth reading."""
        return min(self.entries, key=lambda e: e.free, default=None)


def human_bytes(count: int) -> str:
    """``1536`` -> ``"1.5 KB"``. Two significant places, never a fake one."""
    value = float(max(0, int(count)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def process_rss() -> int:
    """This process's resident set size in bytes, or ``0`` if unknowable.

    ``psutil`` when it is installed, ``/proc/self/statm`` when it is not.
    Zero means "could not measure", and a cleanup that could not measure
    says so rather than reporting a freed figure it made up.
    """
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass
    try:
        with open("/proc/self/statm", "r") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0


def _torch_if_loaded():
    """``torch``, but only if something already imported it.

    Importing torch costs seconds and hundreds of megabytes. A button whose
    job is to *free* memory must not be the thing that allocates it, and a
    process that has not imported torch has certainly not reserved any VRAM
    through it.
    """
    import sys
    return sys.modules.get("torch")


def cuda_reserved() -> Optional[int]:
    """Bytes the CUDA caching allocator holds for this process, or ``None``.

    ``None`` for "there is nothing to ask": no torch, no CUDA build, no
    device, or a CUDA context this process has never initialised. Asking
    would *create* that context — several hundred MB of VRAM — which is a
    strange thing for a button called "clear VRAM" to do.
    """
    torch = _torch_if_loaded()
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return None
        return int(torch.cuda.memory_reserved())
    except Exception:
        LOG.debug("could not read CUDA reserved memory", exc_info=True)
        return None


def _thread_count() -> int:
    """Threads this process is running.

    ``threading.active_count()`` and not ``QThreadPool.activeThreadCount()``:
    the pool reports how many of its threads are *busy*, which goes down
    when work finishes rather than when capacity is retired, and the number
    this button is about is how many threads spaCR is holding open.
    """
    import threading
    return int(threading.active_count())


# ---------------------------------------------------------------------------
# RAM
# ---------------------------------------------------------------------------

def _clear_lru_caches() -> List[str]:
    """Drop every ``lru_cache`` in spaCR's own asset modules."""
    import sys
    done: List[str] = []
    for name in _LRU_CACHE_MODULES:
        module = sys.modules.get(name)
        if module is None:
            # Not imported means not populated. Importing it to clear it
            # would allocate rather than free.
            continue
        for attr in dir(module):
            try:
                value = getattr(module, attr)
            except Exception:
                continue
            clear = getattr(value, "cache_clear", None)
            info = getattr(value, "cache_info", None)
            if not callable(clear) or not callable(info):
                continue
            try:
                held = int(info().currsize)
                if not held:
                    continue
                clear()
                done.append(f"{name}.{attr} ({held} entries)")
            except Exception:
                LOG.debug("could not clear %s.%s", name, attr, exc_info=True)
    return done


def _clear_dict_caches() -> List[str]:
    import sys
    done: List[str] = []
    for module_name, attribute in _DICT_CACHES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        cache = getattr(module, attribute, None)
        if not isinstance(cache, dict) or not cache:
            continue
        held = len(cache)
        try:
            cache.clear()
            done.append(f"{module_name}.{attribute} ({held} entries)")
        except Exception:
            LOG.debug("could not clear %s.%s", module_name, attribute,
                      exc_info=True)
    return done


def _clear_thumbnail_caches() -> List[str]:
    """Empty every live :class:`CropThumbnails` LRU.

    Found by walking spaCR's own widgets rather than by ``gc.get_objects()``:
    a thumbnail cache that is not attached to a screen is already garbage,
    and walking the whole heap to find one would cost more than it frees.
    """
    import sys
    done: List[str] = []
    thumbs_module = sys.modules.get("spacr.qt.crop_thumbs")
    widgets_module = sys.modules.get("PySide6.QtWidgets")
    if thumbs_module is None or widgets_module is None:
        return done
    cls = getattr(thumbs_module, "CropThumbnails", None)
    app = getattr(widgets_module, "QApplication").instance()
    if cls is None or app is None:
        return done
    cleared = 0
    entries = 0
    for widget in list(app.allWidgets()):
        for attr in ("_thumbs", "_thumbnails", "thumbs"):
            cache = getattr(widget, attr, None)
            if not isinstance(cache, cls):
                continue
            try:
                held = len(cache)
                if not held:
                    continue
                cache.clear()
                cleared += 1
                entries += held
            except Exception:
                LOG.debug("could not clear a thumbnail cache", exc_info=True)
    if cleared:
        done.append(f"{cleared} thumbnail cache(s), {entries} thumbnails")
    return done


def _clear_pixmap_cache() -> List[str]:
    """Qt's own pixmap cache — spaCR's process, spaCR's memory."""
    try:
        from PySide6.QtGui import QPixmapCache
        held = int(QPixmapCache.totalUsed())
        if not held:
            return []
        QPixmapCache.clear()
        return [f"Qt pixmap cache ({held} KB)"]
    except Exception:
        LOG.debug("could not clear the Qt pixmap cache", exc_info=True)
        return []


def clear_ram(*, aggressive: bool = False) -> Reclaim:
    """Drop spaCR's own caches and collect. Measured RSS before and after.

    :param aggressive: also drop the caches that are expensive to rebuild
        (thumbnails, icon pixmaps). The mild form keeps them, because a
        cleanup that costs the next screen a second of redrawing is not a
        cleanup, it is a stutter with good intentions.
    :returns: a :class:`Reclaim`. ``freed`` is ``before - after``, from
        :func:`process_rss` — not the size of what was dropped, which would
        be a guess about an allocator nobody here controls.
    """
    before = process_rss()
    details: List[str] = []
    details.extend(_clear_dict_caches())
    if aggressive:
        details.extend(_clear_lru_caches())
        details.extend(_clear_thumbnail_caches())
        details.extend(_clear_pixmap_cache())
    collected = gc.collect()
    if collected:
        details.append(f"{collected} unreachable objects collected")
    after = process_rss()
    note = ""
    if not before or not after:
        return Reclaim("ram", before, after, tuple(details), measured=False,
                       note="this process's memory use could not be read")
    if not details:
        note = ("Nothing was cached, so there was nothing to drop.")
    elif before <= after:
        note = ("The caches are gone; the allocator has not handed those "
                "pages back to the OS, so the process size did not move.")
    return Reclaim("ram", before, after, tuple(details), note=note)


# ---------------------------------------------------------------------------
# VRAM
# ---------------------------------------------------------------------------

def clear_vram(*, release_models: bool = True) -> Reclaim:
    """Return torch's reserved-but-unused CUDA blocks to the driver.

    :param release_models: also run :data:`MODEL_RELEASERS`. Set ``False``
        immediately before a run — see :func:`run_pre_run_cleanup` for why
        releasing a model spaCR is about to reload is a slowdown wearing an
        optimisation's clothes.

    **This cannot reclaim another process's VRAM.** Nothing can: CUDA
    memory belongs to the context that allocated it, and the only way to
    take it back would be to kill that process, which this module does not
    do to anybody. Nor does it touch tensors a running job still holds —
    ``empty_cache()`` frees blocks the allocator is caching, never live
    ones, which is exactly why it is safe to call while work is in flight.
    """
    before = cuda_reserved()
    if before is None:
        torch = _torch_if_loaded()
        if torch is None:
            why = "torch is not loaded in this process, so it holds no VRAM"
        else:
            why = ("this process has no initialised CUDA context, so it "
                   "holds no VRAM")
        return Reclaim("vram", 0, 0, (), note=why, measured=False)

    details: List[str] = []
    if release_models:
        released = 0
        for releaser in list(MODEL_RELEASERS):
            try:
                released += int(releaser() or 0)
            except Exception:
                LOG.debug("a model releaser failed", exc_info=True)
        if released:
            details.append(f"{released} model reference(s) released")
    torch = _torch_if_loaded()
    try:
        torch.cuda.empty_cache()
        details.append("torch.cuda.empty_cache()")
    except Exception:
        LOG.debug("empty_cache failed", exc_info=True)
    after = cuda_reserved()
    after = before if after is None else after
    note = "It cannot reclaim VRAM held by another process — nothing can."
    if not release_models:
        note = ("Model references were kept: this ran immediately before a "
                "run that is about to use them. " + note)
    return Reclaim("vram", before, after, tuple(details), note=note)


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------

#: Never fewer than this many threads for spaCR's own libraries. One is a
#: hang waiting to happen in anything that fans out and joins.
MIN_LIBRARY_THREADS = 2


def _retire_idle_pool_threads() -> List[str]:
    """Let Qt's global pool drop the threads it is not using.

    ``QThreadPool.clear()`` is NOT used and must not be: it discards
    *queued* runnables, which is spaCR's own pending work, not idle
    capacity. Shortening the expiry timeout retires threads that have
    finished and are sitting idle, and leaves everything that is running or
    waiting to run exactly where it is.
    """
    try:
        from PySide6.QtCore import QThreadPool
    except Exception:
        return []
    pool = QThreadPool.globalInstance()
    if pool is None:
        return []
    try:
        previous = int(pool.expiryTimeout())
        pool.setExpiryTimeout(0)
        pool.setExpiryTimeout(previous if previous > 0 else 30000)
        return [f"Qt thread pool: idle threads retired "
                f"({pool.activeThreadCount()} still working)"]
    except Exception:
        LOG.debug("could not retire idle pool threads", exc_info=True)
        return []


def _lower_library_threads(target: Optional[int] = None) -> List[str]:
    """Lower torch's and OpenCV's thread counts — spaCR's own settings."""
    import sys
    done: List[str] = []
    wanted = MIN_LIBRARY_THREADS if target is None else max(
        MIN_LIBRARY_THREADS, int(target))
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            current = int(torch.get_num_threads())
            if current > wanted:
                torch.set_num_threads(wanted)
                done.append(f"torch threads {current} → {wanted}")
        except Exception:
            LOG.debug("could not lower torch threads", exc_info=True)
    cv2 = sys.modules.get("cv2")
    if cv2 is not None:
        try:
            current = int(cv2.getNumThreads())
            if current > wanted:
                cv2.setNumThreads(wanted)
                done.append(f"OpenCV threads {current} → {wanted}")
        except Exception:
            LOG.debug("could not lower OpenCV threads", exc_info=True)
    return done


def clear_cpu(*, target_threads: Optional[int] = None) -> Reclaim:
    """Retire spaCR's own idle workers and lower its thread counts.

    Reads :func:`spacr.qt.bridge.registry` to say what is still running; it
    never cancels it. A parked thread — one that would not stop when its
    owner went away — is *released* here only once it has actually exited;
    :func:`spacr.qt.bridge.prune_parked_threads` is the whole mechanism and
    it waits rather than terminating, because ``QThread.terminate()`` on a
    thread running Python leaves either a held GIL or a corrupt heap.
    """
    before = _thread_count()
    details: List[str] = []
    still_parked = None
    try:
        from .bridge import parked_thread_count, prune_parked_threads, registry
        parked_before = parked_thread_count()
        still_parked = prune_parked_threads()
        if parked_before != still_parked:
            details.append(f"{parked_before - still_parked} parked thread(s) "
                           "released")
        active = len(registry().active())
    except Exception:
        LOG.debug("the run registry is not available", exc_info=True)
        active = 0
    details.extend(_retire_idle_pool_threads())
    details.extend(_lower_library_threads(target_threads))
    after = _thread_count()
    notes = []
    if active:
        notes.append(f"{active} spaCR job(s) are still running and were left "
                     "alone.")
    if still_parked:
        notes.append(f"{still_parked} thread(s) have not exited yet; they are "
                     "parked, not terminated, and will be released when they "
                     "finish.")
    if before == after and not details:
        notes.append("There was no idle capacity to retire.")
    return Reclaim("cpu", before, after, tuple(details), note=" ".join(notes))


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------

def project_paths() -> List[str]:
    """Folders the current project actually touches, most relevant first.

    The source folders the user last pointed each module at, plus the two
    places spaCR writes regardless of where the data lives: the home
    directory (settings, logs, model downloads) and the temp directory.
    Only paths that exist are returned.
    """
    import tempfile
    paths: List[str] = []

    def _add(value) -> None:
        text = str(value or "").strip()
        if not text:
            return
        try:
            resolved = os.path.abspath(os.path.expanduser(text))
        except Exception:
            return
        if os.path.isdir(resolved) and resolved not in paths:
            paths.append(resolved)

    try:
        from .app import APPS
        from .prefs import get_last_source, get_recent_sources
        for row in APPS:
            _add(get_last_source(row[0]))
            for recent in get_recent_sources(row[0], limit=3):
                _add(recent)
    except Exception:
        LOG.debug("could not read the recent project folders", exc_info=True)
    _add(os.path.expanduser("~"))
    try:
        _add(tempfile.gettempdir())
    except Exception:
        pass
    return paths


def disk_report(paths: Optional[Sequence[str]] = None) -> DiskReport:
    """Free space on every drive the project touches. Reads; frees nothing.

    Deduplicated by device id, so a project folder and a home directory on
    the same disk are one line rather than two identical ones — that
    duplication is what makes a disk readout stop being read.
    """
    wanted = list(project_paths() if paths is None else paths)
    entries: List[DiskEntry] = []
    seen_devices = set()
    unreadable = 0
    for path in wanted:
        try:
            device = os.stat(path).st_dev
        except OSError:
            unreadable += 1
            continue
        if device in seen_devices:
            continue
        try:
            usage = shutil.disk_usage(path)
        except OSError:
            unreadable += 1
            continue
        seen_devices.add(device)
        entries.append(DiskEntry(path, int(usage.total), int(usage.used),
                                 int(usage.free)))
    note = ""
    if unreadable:
        note = f"{unreadable} folder(s) could not be read."
    if not entries and not note:
        note = ("No project folder is known yet — open a module and choose a "
                "source folder, and this will report that drive.")
    return DiskReport(tuple(entries), note)


# ---------------------------------------------------------------------------
# What the confirmation says
# ---------------------------------------------------------------------------
# A confirmation that asks "are you sure?" is not a confirmation: a user
# cannot consent to an unnamed action. Each of these names what will happen,
# in the order it will happen, and says what the action cannot do.

_CONFIRMATIONS: Dict[str, Tuple[str, str]] = {
    "ram": (
        "Clear RAM",
        "spaCR will:\n"
        "  • drop its own caches — merged image fields, file-format lookups, "
        "thumbnails, icon and preview pixmaps;\n"
        "  • run a full garbage collection.\n\n"
        "It will not touch any other program, and it will not drop the "
        "operating system's page cache. Cached images are read from disk "
        "again the next time a screen needs them, so the next preview will "
        "be slower.\n\n"
        "You will be told how much was actually freed, measured before and "
        "after."
    ),
    "vram": (
        "Clear VRAM",
        "spaCR will:\n"
        "  • release any model it is still holding;\n"
        "  • call torch.cuda.empty_cache(), returning the GPU blocks torch "
        "has reserved but is not using.\n\n"
        "It cannot reclaim VRAM held by another process — no program can — "
        "and it will not disturb memory a running spaCR job is using. If no "
        "GPU work has happened in this session there is nothing to free, and "
        "it will say so."
    ),
    "cpu": (
        "Clear CPU",
        "spaCR will:\n"
        "  • release its own worker threads that have already finished;\n"
        "  • let Qt retire idle threads from its pool;\n"
        "  • lower its torch and OpenCV thread counts.\n\n"
        "No process is killed and no running or queued job is stopped — "
        "not spaCR's, and certainly not anybody else's work on this "
        "machine. Threads still doing work are left alone.\n\n"
        "It cannot make anything that is already running go faster; it "
        "gives back capacity spaCR is holding and not using."
    ),
    "disk": (
        "Check disk space",
        "spaCR will read the free space on every drive this project "
        "touches — the source folders the modules last used, your home "
        "directory and the temporary directory — and show one line per "
        "drive.\n\n"
        "Nothing is deleted, moved or written. This action only reads."
    ),
}


def confirmation_title(action: str) -> str:
    """The title of ``action``'s confirmation dialog."""
    return _CONFIRMATIONS[action][0]


def confirmation_text(action: str) -> str:
    """What ``action`` will actually do, in words, before it does it."""
    return _CONFIRMATIONS[action][1]


# ---------------------------------------------------------------------------
# The modes
# ---------------------------------------------------------------------------

def _mode() -> str:
    try:
        from .preferences import get_spacr_mode
        return get_spacr_mode()
    except Exception:
        return "balanced"


def _report(result, prefix: str = "") -> None:
    """Log a cleanup result at the level its NEWS VALUE deserves.

    Extra-performance mode runs the pre-run cleanup before every job, and
    most of the time there is nothing to reclaim -- the caches are already
    empty, or the allocator has not handed the pages back. Logging that at
    INFO produced two lines every couple of seconds, all of them saying
    nothing happened (issue #83), which drowns the console and trains the
    reader to ignore it.

    So INFO is reserved for a cleanup that actually moved something, or one
    that found the process LARGER than before -- both are worth a line.
    Everything else is DEBUG, where it is still available when someone is
    diagnosing memory behaviour on purpose.
    """
    worth_saying = result.freed or result.grew
    (LOG.info if worth_saying else LOG.debug)("%s%s", prefix, result.summary())


def _cleanup(*, aggressive: bool, release_models: bool) -> List[Reclaim]:
    results = [clear_ram(aggressive=aggressive),
               clear_vram(release_models=release_models)]
    if aggressive:
        results.append(clear_cpu())
    for result in results:
        _report(result)
    return results


def run_launch_cleanup() -> List[Reclaim]:
    """The cleanup the mode asks for at launch.

    * Extra Performance — everything, models included: nothing is running
      yet, so nothing can be taken out from under a job.
    * Performance — RAM and VRAM, gently.
    * Balanced — nothing at all. Returns ``[]`` without measuring anything,
      because a "cleanup" that measures is still a pause.
    """
    mode = _mode()
    if mode == "extra_performance":
        return _cleanup(aggressive=True, release_models=True)
    if mode == "performance":
        return _cleanup(aggressive=False, release_models=True)
    return []


def run_pre_run_cleanup(app_key: str = "") -> List[Reclaim]:
    """The cleanup Extra Performance runs immediately before a module run.

    Only Extra Performance does this, and it is deliberately **not** the
    same cleanup as at launch:

    * ``release_models=False``. Releasing a model the run is about to reload
      is a slowdown dressed as an optimisation — the reclaim is temporary,
      the reload is seconds of disk and PCIe, and the peak memory is the
      same either way. ``empty_cache()`` still runs, because returning
      *reserved but unused* blocks is exactly what a run about to allocate
      wants.
    * It does not run at all while another run is in flight. The caches this
      drops are the ones a running job is reading, and a cleanup that
      competes with the work it is supposed to be helping is worse than no
      cleanup.
    * It does not touch the CPU: lowering thread counts a moment before a
      run that wants those threads would slow down the very run it precedes.
    """
    if _mode() != "extra_performance":
        return []
    try:
        from .bridge import registry
        if len(registry().active()) > 1:
            LOG.debug("skipping the pre-run cleanup: another run is active")
            return []
    except Exception:
        LOG.debug("could not consult the run registry", exc_info=True)
    results = [clear_ram(aggressive=True), clear_vram(release_models=False)]
    for result in results:
        _report(result, prefix=f"before {app_key or 'a run'}: ")
    return results


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

_INSTALLED = False
_LAUNCH_DONE = False
_SEEN_RUNS: set = set()


def _on_registry_changed() -> None:
    """A run appeared (or finished). Clean up before a NEW one starts.

    ``RunRegistry.changed`` is emitted from inside ``register()``, which
    ``make_thread`` calls *before* it hands the unstarted thread back to its
    caller — so this really does run before the worker does, without a hook
    inside ``bridge.py``.
    """
    try:
        from .bridge import registry
        handles = registry().active()
    except Exception:
        return
    live = {id(handle) for handle in handles}
    _SEEN_RUNS.intersection_update(live)
    fresh = [handle for handle in handles if id(handle) not in _SEEN_RUNS]
    if not fresh:
        return
    _SEEN_RUNS.update(id(handle) for handle in fresh)
    try:
        run_pre_run_cleanup(getattr(fresh[0], "app_key", ""))
    except Exception:
        LOG.debug("the pre-run cleanup failed", exc_info=True)


def install_run_hook() -> bool:
    """Connect the pre-run cleanup to the run registry. Idempotent.

    Nothing in ``bridge.py`` knows this exists: the registry already emits
    when a job is registered, and a mode that is not Extra Performance turns
    the slot into a dictionary lookup and a return.
    """
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        from .bridge import registry
        registry().changed.connect(_on_registry_changed)
    except Exception:
        LOG.debug("could not install the pre-run cleanup hook", exc_info=True)
        return False
    _INSTALLED = True
    return True


def register() -> bool:
    """Entry point for :data:`spacr.qt.SELF_REGISTERING_MODULES`.

    Installs the pre-run hook and performs the launch cleanup the mode asks
    for — once per process, so the test suite calling the launch sequence
    forty times does not collect forty times.
    """
    global _LAUNCH_DONE
    install_run_hook()
    if not _LAUNCH_DONE:
        _LAUNCH_DONE = True
        try:
            run_launch_cleanup()
        except Exception:
            LOG.debug("the launch cleanup failed", exc_info=True)
    return True
