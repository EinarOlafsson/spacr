"""Cache eviction policies and host-memory availability measurements.

The idle timeout and cache ceiling determine the keys selected by
``what_to_drop``. The separate ``headroom_is_short`` predicate compares
available host memory with a configured threshold. Callers apply these
policies and perform resource cleanup.

This module does not unload Python modules or native libraries. Eviction
plans identify cache entries; callers decide which resources may be
released. The policies do not guarantee a particular reduction in process
memory or GPU allocations.
"""
from __future__ import annotations

import logging
from typing import Optional

LOG = logging.getLogger("spacr.qt.memory_budget")

#: Minutes an unused cache entry may sit before it is dropped. 0 means
#: "drop as soon as nothing is using it".
DEFAULT_IDLE_MINUTES: float = 15.0
MIN_IDLE_MINUTES: float = 0.0
MAX_IDLE_MINUTES: float = 480.0

#: Megabytes of cache spaCR may hold at once, across images, merged frames
#: and model weights.
DEFAULT_CACHE_CEILING_MB: int = 2048
MIN_CACHE_CEILING_MB: int = 128
MAX_CACHE_CEILING_MB: int = 131072

#: Megabytes that must stay free for everything else on the machine.
DEFAULT_HEADROOM_MB: int = 2048
MIN_HEADROOM_MB: int = 256
MAX_HEADROOM_MB: int = 131072

#: What each performance level suggests, as
#: ``level -> (idle minutes, cache MB, headroom MB)``.
#:
#: THE TOOLTIPS QUOTE THESE, because each level carries a system
#: configuration recommendation and a number without the
#: machine it suits is not a recommendation. They are suggestions and not
#: overrides: a user who sets a value keeps it.
RECOMMENDED = {
    "laptop": (2.0, 256, 1024),
    "extra_performance": (5.0, 512, 1536),
    "performance": (10.0, 1024, 2048),
    "balanced": (15.0, 2048, 2048),
    "workstation": (60.0, 16384, 8192),
}

#: What each level is FOR, in the words the tooltip uses.
HARDWARE_NOTES = {
    "laptop": "8 GB or less, or running on battery",
    "extra_performance": "a shared machine you do not want spaCR to crowd",
    "performance": "a machine with other work on it",
    "balanced": "an ordinary desktop, 16 GB or more",
    "workstation": "64 GB or more, and yours alone",
}


def recommended_for(level: str):
    """The suggested budget for a performance level.

    :param level: one of `spacr.qt.preferences.PERFORMANCE_LEVELS`.
    :returns: ``(idle_minutes, cache_mb, headroom_mb)``.
    """
    return RECOMMENDED.get(str(level), RECOMMENDED["balanced"])


def free_megabytes() -> Optional[float]:
    """Return available host memory in mebibytes.

    :returns: ``psutil.virtual_memory().available`` divided by ``1024**2``,
        or ``None`` when the measurement fails.
    """
    try:
        import psutil

        return float(psutil.virtual_memory().available) / (1024.0 * 1024.0)
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not read available memory", exc_info=True)
        return None


def headroom_is_short(floor_mb: Optional[float] = None) -> bool:
    """Test whether available host memory is below the configured threshold.

    :param floor_mb: threshold in mebibytes; read from preferences when omitted.
    :returns: ``False`` when memory or the required preference cannot be read;
        otherwise the result of comparing available memory with the threshold.
    """
    free = free_megabytes()
    if free is None:
        return False
    if floor_mb is None:
        try:
            from .preferences import get_headroom_mb

            floor_mb = get_headroom_mb()
        except Exception:                                    # noqa: BLE001
            return False
    return free < float(floor_mb)


def what_to_drop(entries, now: float, idle_minutes: Optional[float] = None,
                 ceiling_mb: Optional[int] = None) -> list:
    """Select cache keys by idle timeout and then the remaining total size.

    :param entries: ``[(key, megabytes, last_used_epoch_seconds), ...]``.
    :param now: the current epoch time, passed in so a test can choose it.
    :param idle_minutes: the idle timeout; from preferences when omitted.
    :param ceiling_mb: the size ceiling; from preferences when omitted.
    :returns: string keys to drop, in selection order.

    Entries older than the timeout are selected first, in input order. If
    the remaining total exceeds the ceiling, additional entries are selected
    from least to most recently used until that total fits.

    Size determines whether further eviction is needed, not the ranking.
    The ceiling can select entries even when none exceeds the idle timeout.
    This helper does not check host-memory headroom or whether entries are
    in use; callers must supply entries eligible for eviction.
    """
    if idle_minutes is None or ceiling_mb is None:
        try:
            from .preferences import get_cache_ceiling_mb, get_idle_minutes

            if idle_minutes is None:
                idle_minutes = get_idle_minutes()
            if ceiling_mb is None:
                ceiling_mb = get_cache_ceiling_mb()
        except Exception:                                    # noqa: BLE001
            idle_minutes = DEFAULT_IDLE_MINUTES if idle_minutes is None \
                else idle_minutes
            ceiling_mb = DEFAULT_CACHE_CEILING_MB if ceiling_mb is None \
                else ceiling_mb

    rows = [(str(key), float(size), float(used)) for key, size, used in entries]
    cutoff = float(now) - float(idle_minutes) * 60.0
    doomed = [key for key, _size, used in rows if used < cutoff]

    kept = [row for row in rows if row[0] not in set(doomed)]
    total = sum(size for _key, size, _used in kept)
    if total > float(ceiling_mb):
        for key, size, _used in sorted(kept, key=lambda row: row[2]):
            if total <= float(ceiling_mb):
                break
            doomed.append(key)
            total -= size
    return doomed
