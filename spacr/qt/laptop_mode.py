"""Measure hardware constraints and suppress the ambient backdrop for this process.

Hardware-based decisions use the usable CPU count, physical memory and the
``SPACR_LAPTOP_MODE`` override. GUI startup instead passes an explicit decision
from the selected performance level; its hardware reading is diagnostic.

Suppression affects backdrop rendering. This module does not set worker-thread
limits or analysis parameters, and it does not replace stored animation
preferences.

``apply`` sets ``SPACR_NO_BACKDROP`` when suppression is requested and the
ambient backdrop is enabled. Disabling removes only a suppression created by
this module; a pre-existing crash-recovery override is retained.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

#: Below this many usable cores, a machine is treated as small. Four is the
#: line because the ambient layer, the blur and the fit all want a core each
#: and the interface still needs one to stay answerable.
SMALL_CORE_COUNT = 4

#: Below this much RAM in GiB, likewise. 8 GiB is where a browser, a Python
#: process holding a plate of images and the compositor stop fitting.
SMALL_MEMORY_GIB = 8.0

#: The environment variable that overrides the decision either way.
#: ``1``/``true``/``on`` forces it on, ``0``/``false``/``off`` forces it off,
#: absent leaves it to the measurement.
OVERRIDE_VARIABLE = "SPACR_LAPTOP_MODE"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def usable_cores() -> int:
    """Return the CPU affinity count, falling back to ``os.cpu_count``.

    ``os.sched_getaffinity`` accounts for process pinning when available. If
    it is unavailable or fails, use the reported CPU count instead. The
    result is at least one; the fallback does not measure affinity limits.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):        # pragma: no cover - not Linux
        return max(1, os.cpu_count() or 1)


def total_memory_gib() -> Optional[float]:
    """Physical memory in GiB, or None when it cannot be read.

    Windows has no ``os.sysconf``; psutil reads the same figure there.
    """
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        return (pages * size) / (1024 ** 3)
    except (AttributeError, ValueError, OSError):
        pass
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return None


def override() -> Optional[bool]:
    """Parse the environment override; return ``None`` if absent or unrecognized."""
    raw = str(os.environ.get(OVERRIDE_VARIABLE, "")).strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return None


def measure() -> Dict[str, object]:
    """Return CPU count, physical memory, threshold flags and the parsed override."""
    cores = usable_cores()
    memory = total_memory_gib()
    return {
        "cores": cores,
        "memory_gib": memory,
        "few_cores": cores < SMALL_CORE_COUNT,
        "little_memory": memory is not None and memory < SMALL_MEMORY_GIB,
        "override": override(),
    }


def wanted(reading: Optional[Dict[str, object]] = None) -> Tuple[bool, str]:
    """Return the hardware-based laptop-mode decision and its explanation.

    :returns: ``(on, why)``. The reason is always populated, including when
        the decision is off. An explicit environment override takes
        precedence over the core-count and physical-memory threshold flags.
    """
    reading = measure() if reading is None else reading
    chosen = reading.get("override")
    cores = reading.get("cores")
    memory = reading.get("memory_gib")
    seen = (f"{cores} usable core(s)"
            + (f", {memory:.1f} GiB of memory" if memory else ""))

    if chosen is True:
        return True, f"laptop mode: on because {OVERRIDE_VARIABLE} asks for it"
    if chosen is False:
        return False, f"laptop mode: off because {OVERRIDE_VARIABLE} asks for it"

    reasons: List[str] = []
    if reading.get("few_cores"):
        reasons.append(f"fewer than {SMALL_CORE_COUNT} usable cores")
    if reading.get("little_memory"):
        reasons.append(f"less than {SMALL_MEMORY_GIB:g} GiB of memory")
    if reasons:
        return True, (f"laptop mode: on -- this machine has {seen} "
                      f"({' and '.join(reasons)}). "
                      f"Set {OVERRIDE_VARIABLE}=0 to keep everything on.")
    return False, (f"laptop mode: off -- this machine has {seen}. "
                   f"Set {OVERRIDE_VARIABLE}=1 to turn it on anyway.")


def what_it_turns_down() -> Tuple[Tuple[str, str], ...]:
    """Return the animation and backdrop-blur descriptions used in diagnostics.

    Each pair contains an effect name and its description for UI notes and
    logs. This function returns text; :func:`apply` performs the process-local
    backdrop suppression.
    """
    return (
        ("the ambient background animation",
         "the moving backdrop stops; every colour and control is unchanged"),
        ("the backdrop blur",
         "panels sit on a flat ground instead of a blurred one"),
    )


def describe() -> str:
    """Format the hardware-based decision and effect descriptions for diagnostic logs."""
    on, why = wanted()
    lines = [why]
    if on:
        lines.append("turned down, and only the drawing:")
        lines.extend(f"  - {what}: {cost}" for what, cost in what_it_turns_down())
        lines.append("a run computes exactly the same answer either way.")
    return "\n".join(lines)


#: The process-local backdrop switch, and the only thing this module turns.
#:
#: SUPPRESSING IS NOT THE SAME AS DECIDING, and this module used to do the
#: second. It called `preferences.set_ambient_enabled(False)`, which is the
#: USER'S stored answer -- so on a machine under the core or memory bar,
#: every launch overwrote a preference the user had set, and a user who
#: turned the animation back on got it taken away again the next time they
#: started spaCR, with nothing on the surface to say why. A hardware guess
#: was writing into the one row that is supposed to mean "what I chose".
#:
#: `SPACR_NO_BACKDROP` is the mechanism that already existed for exactly
#: this, and `get_ambient_enabled` documents it: process-local, never saved,
#: honoured before the stored value, "so the next clean run brings it back
#: with nothing for the user to undo". `crash_recovery` uses it after spaCR
#: has died on launch twice. A performance constraint is the same shape of
#: thing -- true for this run, on this machine, and none of it the user's
#: opinion.
#:
#: Found by the Preferences audit.
_NO_BACKDROP = "SPACR_NO_BACKDROP"

#: Whether the variable above is OURS to clear.
#:
#: `crash_recovery` sets the same variable, and it sets it because spaCR has
#: already failed to start twice. Turning the mode off must not hand the
#: backdrop back to a driver that has crashed on it -- so `apply` clears the
#: variable only when this module is what set it.
_suppressed_here = False


def apply(on: Optional[bool] = None) -> Dict[str, object]:
    """Apply process-local backdrop suppression and report the decision and actions.

    :param on: explicit decision; ``None`` uses :func:`wanted`, including its
        hardware reading and environment override.
    :returns: a mapping with ``on``, ``why`` and ``changed``. ``on`` is the
        decision; ``changed`` lists actions actually performed and may be empty.

    Suppression is process-local and does not replace the stored animation
    preference. Disabling clears only suppression created by this module,
    retaining any pre-existing crash-recovery override.
    """
    global _suppressed_here

    if on is None:
        on, why = wanted()
    else:
        why = (f"laptop mode: {'on' if on else 'off'} because the caller "
               f"asked for it")

    changed: List[str] = []
    if on:
        try:
            from . import preferences

            if preferences.get_ambient_enabled():
                os.environ[_NO_BACKDROP] = "1"
                _suppressed_here = True
                changed.append("ambient backdrop suppressed for this run")
        except Exception:                                # noqa: BLE001
            pass
    elif _suppressed_here:
        os.environ.pop(_NO_BACKDROP, None)
        _suppressed_here = False
        changed.append("ambient backdrop restored")
    return {"on": on, "why": why, "changed": changed}
