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
    Drops spaCR's own caches: the merged-field LRU
    (:func:`spacr.crops.clear_field_cache` — by far the largest, whole image
    stacks), the file-format and DB-format caches, the zoomed-animation
    cache, the icon and preview ``lru_cache``s, the filter-kind cache, every
    live :class:`~spacr.qt.crop_thumbs.CropThumbnails` thumbnail LRU, and
    Qt's own ``QPixmapCache``. A process with no live Qt application also
    runs ``gc.collect()``. The GUI deliberately does not: walking a heap of
    live Qt wrappers can enter already-retired C++ objects and crash the
    process. It cannot return memory the allocator has decided to keep, and
    it says so when the measured RSS does not move.

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
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

LOG = logging.getLogger("spacr.qt.resource_cleanup")

__all__ = [
    "Reclaim", "DiskEntry", "DiskReport", "BudgetSweep",
    "clear_ram", "clear_vram", "clear_cpu", "disk_report",
    "confirmation_title", "confirmation_text", "ACTIONS",
    "MODEL_RELEASERS", "process_rss", "cuda_reserved",
    "run_launch_cleanup", "run_pre_run_cleanup", "install_run_hook",
    "register", "sweep_memory_budget", "install_budget_sweep",
]

#: The four buttons, in the order Preferences shows them.
ACTIONS: Tuple[str, ...] = ("ram", "vram", "cpu", "disk")

#: Callables that release a model reference spaCR itself is holding, each
#: returning how many it released.  Most pipeline models are run-scoped.  A
#: screen that deliberately keeps another warm model registers it here; the
#: built-in Annotate outline model is also discovered from its already-loaded
#: module so cleanup never imports Cellpose merely to ask whether it is warm.
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
    ("spacr.qt.annotate_engine", "_MASK_CACHE"),
    ("spacr.qt.annotate_engine", "_EDGE_CACHE"),
)

# A sweep does bounded work on the GUI thread.  If more is required, the next
# five-second tick continues it; no single pass can pickle/spill hundreds of
# figures and turn a memory safeguard into the event-loop freeze it prevents.
BUDGET_SWEEP_INTERVAL_MS = 5000
BUDGET_SWEEP_MAX_ENTRIES = 64


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


@dataclass(frozen=True)
class BudgetSweep:
    """Observable result of applying the user's live-cache policy.

    ``before_mb``/``after_mb`` are sums of the cache owners' measured entry
    sizes, not process RSS.  That makes the accounting stable and attributable:
    shared pages and Python's allocator cannot make an evicted entry appear to
    have grown.  RSS remains the measurement reported by :class:`Reclaim` for
    the explicit Clear RAM action.
    """

    before_mb: float = 0.0
    after_mb: float = 0.0
    dropped: Tuple[str, ...] = ()
    retained_in_use: Tuple[str, ...] = ()
    pressure: bool = False
    complete: bool = True
    models_released: int = 0
    vram_freed: int = 0
    errors: Tuple[str, ...] = ()

    @property
    def freed_mb(self) -> float:
        """Measured cache bytes removed by this sweep, in MiB."""
        return max(0.0, float(self.before_mb) - float(self.after_mb))


@dataclass(frozen=True)
class _BudgetEntry:
    token: str
    label: str
    megabytes: float
    last_used: float
    in_use: bool
    drop: Callable[[], bool]


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


def _cuda_stat(torch, name: str) -> int:
    """Sum one allocator statistic across already-available CUDA devices."""
    getter = getattr(torch.cuda, name)
    try:
        count = max(1, int(torch.cuda.device_count()))
    except Exception:                                       # noqa: BLE001
        count = 1
    if count == 1:
        return int(getter())
    try:
        return sum(int(getter(device)) for device in range(count))
    except TypeError:
        # Small test doubles and old compatible torch builds may expose only
        # the no-argument form.  It still measures the current device rather
        # than turning a cleanup into an import or context initialisation.
        return int(getter())


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
        return _cuda_stat(torch, "memory_reserved")
    except Exception:
        LOG.debug("could not read CUDA reserved memory", exc_info=True)
        return None


def _cuda_cached() -> Optional[int]:
    """Bytes in torch's loaded CUDA allocator that no tensor is using.

    This is the part :func:`torch.cuda.empty_cache` can honestly return.
    ``memory_reserved - memory_allocated`` deliberately excludes live tensor
    storage, so the budget never calls an allocation "cache" merely because
    torch owns it.  Like :func:`cuda_reserved`, this imports nothing and does
    not initialise a CUDA context.
    """
    torch = _torch_if_loaded()
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return None
        reserved = _cuda_stat(torch, "memory_reserved")
        allocated = _cuda_stat(torch, "memory_allocated")
        return max(0, reserved - allocated)
    except Exception:
        LOG.debug("could not read CUDA cached memory", exc_info=True)
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
# The live-cache budget
# ---------------------------------------------------------------------------

def _loaded_cache_owners():
    """Return cache owners already present in this process; import nothing.

    A cleanup whose purpose is to release memory must never import the screen
    or pipeline that owns it.  Module caches expose the same two-method
    protocol as object caches; widget-owned caches publish weak snapshots, so
    observing them cannot extend their lifetime.
    """
    owners = []
    for module_name in ("spacr.crops", "spacr.qt.annotate_engine"):
        module = sys.modules.get(module_name)
        if (module is not None
                and callable(getattr(module, "cache_budget_entries", None))
                and callable(getattr(module, "drop_cache_budget_entry", None))):
            owners.append(module)
    for module_name, accessor_name in (
            ("spacr.qt.crop_thumbs", "live_thumbnail_caches"),
            ("spacr.qt.widgets.figure_queue", "live_figure_queues"),
            ("spacr.qt.widgets.timelapse_preview", "_live_cache_owners"),
            ("spacr.qt.widgets.timelapse_movie", "_live_cache_owners")):
        module = sys.modules.get(module_name)
        accessor = getattr(module, accessor_name, None)
        if not callable(accessor):
            continue
        try:
            owners.extend(accessor())
        except Exception:                                      # noqa: BLE001
            LOG.debug("could not enumerate %s", module_name, exc_info=True)
    # A buggy screen must not make the same owner count twice.
    return tuple({id(owner): owner for owner in owners}.values())


def _owner_label(owner) -> str:
    return str(getattr(owner, "__name__", "")
               or f"{type(owner).__module__}.{type(owner).__name__}")


def _collect_budget_entries(owners=None):
    records: List[_BudgetEntry] = []
    errors: List[str] = []
    for owner in tuple(_loaded_cache_owners() if owners is None else owners):
        label = _owner_label(owner)
        inventory = (getattr(owner, "cache_budget_entries", None)
                     or getattr(owner, "_cache_budget_entries", None))
        dropper = (getattr(owner, "drop_cache_budget_entry", None)
                   or getattr(owner, "_drop_cache_budget_entry", None))
        if not callable(inventory) or not callable(dropper):
            errors.append(f"{label}: cache-budget protocol is incomplete")
            continue
        try:
            rows = inventory()
        except Exception as exc:                               # noqa: BLE001
            errors.append(f"{label}: inventory failed ({exc})")
            LOG.debug("could not inventory %s", label, exc_info=True)
            continue
        for ordinal, row in enumerate(rows):
            try:
                key, byte_count, last_used, in_use = row
                token = f"cache-{len(records):08d}"
                key_label = repr(key)
                if len(key_label) > 160:
                    key_label = key_label[:157] + "..."

                def _drop(dropper=dropper, key=key):
                    return bool(dropper(key))

                records.append(_BudgetEntry(
                    token=token,
                    label=f"{label}[{key_label}]",
                    megabytes=max(0, int(byte_count)) / (1024.0 * 1024.0),
                    last_used=float(last_used),
                    in_use=bool(in_use),
                    drop=_drop,
                ))
            except Exception as exc:                           # noqa: BLE001
                errors.append(f"{label} entry {ordinal}: invalid ({exc})")
                LOG.debug("invalid cache entry from %s", label, exc_info=True)
    return records, errors


def _budget_values(idle_minutes, ceiling_mb):
    from .memory_budget import (
        DEFAULT_CACHE_CEILING_MB,
        DEFAULT_IDLE_MINUTES,
    )

    if idle_minutes is None or ceiling_mb is None:
        try:
            from .preferences import get_cache_ceiling_mb, get_idle_minutes

            if idle_minutes is None:
                idle_minutes = get_idle_minutes()
            if ceiling_mb is None:
                ceiling_mb = get_cache_ceiling_mb()
        except Exception:                                    # noqa: BLE001
            idle_minutes = (DEFAULT_IDLE_MINUTES
                            if idle_minutes is None else idle_minutes)
            ceiling_mb = (DEFAULT_CACHE_CEILING_MB
                          if ceiling_mb is None else ceiling_mb)
    return max(0.0, float(idle_minutes)), max(0.0, float(ceiling_mb))


def _a_run_is_active() -> bool:
    """Read the already-loaded registry without importing the Qt bridge."""
    bridge = sys.modules.get("spacr.qt.bridge")
    registry = bridge.registry if bridge is not None else None
    if not callable(registry):
        return False
    try:
        return bool(registry().active())
    except Exception:                                        # noqa: BLE001
        return True


# The allocator exposes no "last kernel finished" timestamp.  Observing its
# reclaimable byte count on the existing five-second sweep is the honest
# substitute: a change, or any registered run in flight, is activity; an
# unchanged cache after the run is idle.  These two scalars retain no tensor
# and therefore cannot become another cache themselves.
_CUDA_CACHE_BYTES: Optional[int] = None
_CUDA_CACHE_LAST_USED = 0.0


def _observe_cuda_cache(now: float, *, run_active: bool
                        ) -> Tuple[Optional[int], float]:
    """Return ``(reclaimable bytes, last activity)`` without importing torch."""
    global _CUDA_CACHE_BYTES, _CUDA_CACHE_LAST_USED
    cached = _cuda_cached()
    if cached is None or cached <= 0:
        _CUDA_CACHE_BYTES = cached
        _CUDA_CACHE_LAST_USED = 0.0
        return cached, 0.0
    if (run_active or _CUDA_CACHE_BYTES != cached
            or _CUDA_CACHE_LAST_USED <= 0.0):
        _CUDA_CACHE_LAST_USED = float(now)
    _CUDA_CACHE_BYTES = int(cached)
    return int(cached), float(_CUDA_CACHE_LAST_USED)


def _record_cuda_cleanup(now: float) -> None:
    """Refresh allocator accounting after an attempted policy cleanup."""
    global _CUDA_CACHE_BYTES, _CUDA_CACHE_LAST_USED
    cached = _cuda_cached()
    _CUDA_CACHE_BYTES = cached
    _CUDA_CACHE_LAST_USED = float(now) if cached else 0.0


def _release_models_under_pressure() -> int:
    """Drop registered warm models only when no run can be using them."""
    if _a_run_is_active():
        return 0
    released = 0
    for releaser in _loaded_model_releasers():
        try:
            released += max(0, int(releaser() or 0))
        except Exception:                                    # noqa: BLE001
            LOG.debug("a model releaser failed", exc_info=True)
    return released


def _loaded_model_releasers():
    """Known releasers from loaded modules, with no imports or duplicates."""
    releasers = list(MODEL_RELEASERS)
    for module_name in ("spacr.qt.annotate_engine",):
        module = sys.modules.get(module_name)
        candidate = getattr(module, "_release_cached_models", None)
        if callable(candidate) and candidate not in releasers:
            releasers.append(candidate)
    return tuple(releasers)


def sweep_memory_budget(*, now: Optional[float] = None,
                        idle_minutes: Optional[float] = None,
                        ceiling_mb: Optional[float] = None,
                        headroom_short: Optional[bool] = None,
                        max_entries: int = BUDGET_SWEEP_MAX_ENTRIES,
                        owners=None) -> BudgetSweep:
    """Apply idle age, one global byte ceiling, and the free-memory floor.

    :param now: epoch seconds; explicit so a controlled-clock test can drive
        real cache entries.
    :param idle_minutes: override for tests; otherwise the live preference.
    :param ceiling_mb: global RAM-cache ceiling; otherwise the preference.
    :param headroom_short: controlled pressure state for tests.  When omitted,
        :func:`spacr.qt.memory_budget.headroom_is_short` is measured before
        and after each pressure eviction, stopping as soon as the floor is
        restored.
    :param max_entries: hard bound on evictions in this call.
    :param owners: optional owner sequence for an isolated test; production
        discovers all already-loaded registered caches.
    :returns: measured, attributable accounting for the pass.

    In-use entries are excluded before either policy is evaluated.  Their
    bytes still count against the one process-wide ceiling, so a pinned 100 MB
    Figure leaves 100 MB less room for evictable thumbnails; applying a full
    ceiling independently to every cache would multiply the user's setting by
    the number of open screens.
    """
    from . import memory_budget

    instant = time.time() if now is None else float(now)
    idle, ceiling = _budget_values(idle_minutes, ceiling_mb)
    run_active = _a_run_is_active()
    cuda_bytes, cuda_last_used = _observe_cuda_cache(
        instant, run_active=run_active)
    cuda_due = bool(
        not run_active
        and cuda_bytes
        and ((float(cuda_bytes) / (1024.0 * 1024.0)) > ceiling
             or instant - cuda_last_used >= idle * 60.0)
    )
    entries, errors = _collect_budget_entries(owners)
    before = sum(row.megabytes for row in entries)
    pinned = [row for row in entries if row.in_use]
    candidates = [row for row in entries if not row.in_use]
    pinned_mb = sum(row.megabytes for row in pinned)
    available_ceiling = max(0.0, ceiling - pinned_mb)
    policy_tokens = memory_budget.what_to_drop(
        [(row.token, row.megabytes, row.last_used) for row in candidates],
        instant, idle_minutes=idle, ceiling_mb=available_ceiling)
    by_token = {row.token: row for row in candidates}
    normal = [by_token[token] for token in policy_tokens if token in by_token]

    explicit_pressure = headroom_short is not None
    pressure = (bool(headroom_short) if explicit_pressure
                else memory_budget.headroom_is_short())
    limit = max(0, int(max_entries))
    attempted = set()
    dropped: List[str] = []
    freed = 0.0

    def _evict(row: _BudgetEntry) -> bool:
        nonlocal freed
        attempted.add(row.token)
        try:
            removed = bool(row.drop())
        except Exception as exc:                              # noqa: BLE001
            errors.append(f"{row.label}: eviction failed ({exc})")
            LOG.debug("could not evict %s", row.label, exc_info=True)
            return False
        if removed:
            dropped.append(row.label)
            freed += row.megabytes
        return removed

    for row in normal:
        if len(attempted) >= limit:
            break
        _evict(row)

    # Headroom is a hard floor, not another per-cache size.  Once ordinary
    # idle/ceiling evictions have run, release the coldest remaining entries
    # until the measured floor is restored or this bounded pass is exhausted.
    remaining = sorted((row for row in candidates
                        if row.token not in attempted),
                       key=lambda row: row.last_used)
    pressure_remaining = pressure
    if pressure_remaining and not explicit_pressure:
        pressure_remaining = memory_budget.headroom_is_short()
    while pressure_remaining and remaining and len(attempted) < limit:
        row = remaining.pop(0)
        _evict(row)
        pressure_remaining = (True if explicit_pressure
                              else memory_budget.headroom_is_short())

    models_released = 0
    vram_freed = 0
    allocator_attempted = False
    if (pressure_remaining or cuda_due) and len(attempted) < limit \
            and not run_active:
        allocator_attempted = True
    if pressure_remaining and len(attempted) < limit and not run_active:
        models_released = _release_models_under_pressure()
        # CUDA allocator blocks are reclaimable state too.  This never imports
        # torch or initialises a CUDA context; clear_vram has both guards.
        result = clear_vram(release_models=False)
        vram_freed = result.freed
        _record_cuda_cleanup(instant)
    elif cuda_due and len(attempted) < limit and not run_active:
        # A stable allocator cache obeys the same idle timeout and byte
        # ceiling as the RAM caches.  Live allocations are excluded by
        # ``_cuda_cached`` and a registered run pins the allocator wholesale.
        result = clear_vram(release_models=False)
        vram_freed = result.freed
        _record_cuda_cleanup(instant)

    pending_normal = any(row.token not in attempted for row in normal)
    pending_pressure = bool(pressure_remaining and remaining)
    pending_cuda = bool(cuda_due and not allocator_attempted)
    complete = not pending_normal and not pending_pressure and not pending_cuda
    after = max(0.0, before - freed)
    return BudgetSweep(
        before_mb=before,
        after_mb=after,
        dropped=tuple(dropped),
        retained_in_use=tuple(row.label for row in pinned),
        pressure=pressure,
        complete=complete,
        models_released=models_released,
        vram_freed=vram_freed,
        errors=tuple(errors),
    )


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
            try:
                clear = getattr(value, "cache_clear", None)
                info = getattr(value, "cache_info", None)
            except Exception:
                LOG.debug("could not inspect %s.%s", name, attr,
                          exc_info=True)
                continue
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
            metadata = getattr(module, f"{attribute}_USED", None)
            if isinstance(metadata, dict):
                metadata.clear()
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
    app = widgets_module.QApplication.instance()
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

        # Qt 6 removed ``totalUsed`` from the Python API. Absence of an
        # accounting reading must not turn into absence of the cleanup: clear
        # the cache either way, and report a byte count only when Qt supplied
        # one. Inventing a count would violate Reclaim's measured contract.
        total_used = getattr(QPixmapCache, "totalUsed", None)
        held = int(total_used()) if callable(total_used) else None
        QPixmapCache.clear()
        return [f"Qt pixmap cache ({held} KB)"] if held else []
    except Exception:
        LOG.debug("could not clear the Qt pixmap cache", exc_info=True)
        return []


def _qt_application_is_running() -> bool:
    """Whether a loaded PySide application owns live Qt wrappers.

    Do not import Qt to answer this.  A headless cleanup must stay headless,
    both for launch cost and so it keeps the ordinary ``gc.collect`` path.
    """
    import sys

    widgets = sys.modules.get("PySide6.QtWidgets")
    application = getattr(widgets, "QApplication", None)
    if application is None:
        return False
    try:
        instance = application.instance()
        return isinstance(instance, application)
    except Exception:
        return False


def clear_ram(*, aggressive: bool = False) -> Reclaim:
    """Drop spaCR's own caches, measured by RSS before and after.

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
    qt_is_live = _qt_application_is_running()
    if not qt_is_live:
        collected = gc.collect()
        if collected:
            details.append(f"{collected} unreachable objects collected")
    after = process_rss()
    notes: List[str] = []
    if qt_is_live:
        notes.append(
            "A full Python garbage collection was skipped while Qt widgets "
            "were live.")
    if not before or not after:
        notes.insert(0, "this process's memory use could not be read")
        return Reclaim("ram", before, after, tuple(details), measured=False,
                       note=" ".join(notes))
    if not details:
        notes.insert(0, "Nothing was cached, so there was nothing to drop.")
    elif before <= after:
        notes.insert(
            0, "The caches are gone; the allocator has not handed those "
               "pages back to the OS, so the process size did not move.")
    return Reclaim("ram", before, after, tuple(details),
                   note=" ".join(notes))


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
        for releaser in _loaded_model_releasers():
            try:
                released += int(releaser() or 0)
            except Exception:
                LOG.debug("a model releaser failed", exc_info=True)
        if released:
            details.append(f"{released} model reference(s) released")
    torch = _torch_if_loaded()
    try:
        # EVERY BACKEND CACHES, not only CUDA. Metal holds freed blocks in
        # exactly the same way and answers `torch.mps.empty_cache()`; on a
        # 4 GB card that is the difference between the next screen opening
        # and an allocation failure. Routed through the resolver so the
        # right call is made without a vendor branch here. See 319.
        from ..accelerator import empty_cache as release_device_memory

        release_device_memory()
        details.append("device cache released")
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
        "thumbnails, icon and preview pixmaps.\n\n"
        "It will not force a full Python garbage collection while Qt "
        "widgets are live: collecting those wrappers can crash the "
        "application.\n\n"
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
    """What ``action`` will actually do, in words, before it does it.

    The long form, for the confirmation the user is asked to agree to. A
    bulleted list is right there: they are about to authorise it, and the
    bullets are what they are authorising.
    """
    return _CONFIRMATIONS[action][1]


#: The same promise as :data:`_CONFIRMATIONS`, as one sentence.
#:
#: A HINT BAR IS NOT A CONFIRMATION DIALOG. The long forms are four to eight
#: lines of bulleted text, and the strip under the Preferences tabs grew to
#: fit whichever one the pointer was over -- so moving between two buttons
#: made the dialog jump. The compact form keeps the hint to one paragraph.
#:
#: What is dropped is the enumeration, never the limit: each of these still
#: says what the action will NOT do, because that is the part a user is
#: uneasy about on a shared machine.
_SUMMARIES: Dict[str, str] = {
    "ram": (
        "Drops spaCR's own caches without forcing Python garbage collection "
        "over live Qt widgets. No other program is touched, and the next "
        "preview is slower because its images are read again. You are told "
        "how much was actually freed."
    ),
    "vram": (
        "Releases any model still held and returns the GPU blocks torch has "
        "reserved but is not using. VRAM held by another process cannot be "
        "reclaimed, and memory a running spaCR job is using is left alone."
    ),
    "cpu": (
        "Retires spaCR's finished worker threads and lowers its torch and "
        "OpenCV thread counts. No process is killed and no running or queued "
        "job is stopped; threads still doing work are left alone."
    ),
    "disk": (
        "Reads the free space on every drive this project touches and "
        "reports it. Nothing is written, moved or deleted."
    ),
}


def summary_text(action: str) -> str:
    """One paragraph saying what ``action`` does.

    :param action: ``'ram'``, ``'vram'``, ``'cpu'`` or ``'disk'``.
    :returns: the short form for a hover, falling back to the long form so a
        new action is never left with no help at all.
    """
    return _SUMMARIES.get(action) or _CONFIRMATIONS[action][1]


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
_BUDGET_TIMER = None
_BUDGET_SWEEP_PENDING = False


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
        # A run just finished (or the registry only shed a handle).  Its
        # caches are no longer in use; let the event loop settle, then apply
        # the same global policy the periodic sweep uses.
        _request_budget_sweep()
        return
    _SEEN_RUNS.update(id(handle) for handle in fresh)
    try:
        run_pre_run_cleanup(getattr(fresh[0], "app_key", ""))
    except Exception:
        LOG.debug("the pre-run cleanup failed", exc_info=True)


def _budget_tick() -> None:
    global _BUDGET_SWEEP_PENDING
    _BUDGET_SWEEP_PENDING = False
    try:
        result = sweep_memory_budget()
        if result.dropped or result.models_released or result.vram_freed:
            LOG.info(
                "memory budget: %.1f -> %.1f MiB; %d cache entries, "
                "%d model references and %s VRAM released",
                result.before_mb, result.after_mb, len(result.dropped),
                result.models_released, human_bytes(result.vram_freed))
        elif result.errors:
            LOG.debug("memory budget sweep: %s", "; ".join(result.errors))
    except Exception:                                        # noqa: BLE001
        LOG.debug("the live-cache budget sweep failed", exc_info=True)


def _request_budget_sweep() -> None:
    """Queue a sweep after the current Qt signal/paint has returned."""
    global _BUDGET_SWEEP_PENDING
    if _BUDGET_SWEEP_PENDING:
        return
    _BUDGET_SWEEP_PENDING = True
    try:
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, _budget_tick)
    except Exception:                                        # noqa: BLE001
        _BUDGET_SWEEP_PENDING = False
        # No event loop means the periodic integration is inapplicable.  The
        # explicit ``sweep_memory_budget`` API remains usable by headless code.
        LOG.debug("could not queue the live-cache sweep", exc_info=True)


def install_budget_sweep() -> bool:
    """Install the bounded periodic policy sweep on the live QApplication."""
    global _BUDGET_TIMER
    if _BUDGET_TIMER is not None:
        try:
            if _BUDGET_TIMER.isActive():
                return True
        except RuntimeError:
            _BUDGET_TIMER = None
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            return False
        timer = QTimer(app)
        timer.setObjectName("LiveCacheBudgetSweep")
        timer.setInterval(BUDGET_SWEEP_INTERVAL_MS)
        timer.setSingleShot(False)
        timer.timeout.connect(_budget_tick)
        timer.start()
        _BUDGET_TIMER = timer
        return True
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not install the live-cache budget sweep",
                  exc_info=True)
        return False


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
    install_budget_sweep()
    if not _LAUNCH_DONE:
        _LAUNCH_DONE = True
        try:
            run_launch_cleanup()
        except Exception:
            LOG.debug("the launch cleanup failed", exc_info=True)
    return True
