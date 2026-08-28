"""What spaCR is allowed to keep, and when it has to give it back.

Three settings, and the order matters. The HEADROOM FLOOR comes first: it
says how much of the machine must stay free for everything else, and the
other two are meaningless until something says when to apply them. Then the
IDLE TIMEOUT, which says how long an unused thing may sit before it is
dropped, and the CACHE CEILING, which says how much may be held at all.

WHAT THESE DO NOT DO IS UNLOAD A LIBRARY, and nothing here is named as
though it does. Measured on this machine: importing torch costs 477 MB,
deleting every torch entry from `sys.modules` and collecting returns 0 of
it, and 63 of its shared objects stay mapped. CPython has never supported
unloading a C extension. What CAN be returned is caches, model weights and
GPU allocations, so that is what these govern -- and deferring an import
until first use is the honest form of "load when called", which is why a
session that never opens a deep-learning module never pays the 477 MB.
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
#: THE TOOLTIPS QUOTE THESE, because the maintainer asked for "system
#: configuration recomendations for each level" and a number without the
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
    """How much memory the machine has free right now.

    :returns: megabytes, or ``None`` when it cannot be measured -- in which
        case the headroom floor cannot be enforced and says so rather than
        guessing.
    """
    try:
        import psutil

        return float(psutil.virtual_memory().available) / (1024.0 * 1024.0)
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not read available memory", exc_info=True)
        return None


def headroom_is_short(floor_mb: Optional[float] = None) -> bool:
    """Whether free memory has fallen below the floor.

    :param floor_mb: the floor; read from preferences when omitted.
    :returns: ``False`` when memory cannot be measured -- a cache that
        cannot be shown to be a problem is not dropped on suspicion.
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
    """Which cache entries must go, oldest idle first.

    :param entries: ``[(key, megabytes, last_used_epoch_seconds), ...]``.
    :param now: the current epoch time, passed in so a test can choose it.
    :param idle_minutes: the idle timeout; from preferences when omitted.
    :param ceiling_mb: the size ceiling; from preferences when omitted.
    :returns: the keys to drop, in the order to drop them.

    TWO REASONS, APPLIED IN ORDER. Anything idle longer than the timeout
    goes because nothing is using it. Then, if what remains is still over
    the ceiling, the least recently used go until it fits -- so a cache
    under pressure gives up what it is least likely to want next rather
    than whatever it happens to reach first.

    An entry is never dropped for being large alone: size decides the ORDER
    of a trim, and idleness decides whether one happens.
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
        # Least recently used first, which is the one least likely to be
        # wanted next.
        for key, size, _used in sorted(kept, key=lambda row: row[2]):
            if total <= float(ceiling_mb):
                break
            doomed.append(key)
            total -= size
    return doomed
