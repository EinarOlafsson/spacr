"""Run well on a small machine without becoming a smaller application.

THE RULE THIS FILE ANSWERS TO, in the maintainer's words: "just turning
things of is one strategy, but my best case scenario is being able to keep
as many features as possible just optimizing them so they also run on worse
hardware." So this is the FALLBACK, reached after the optimisations, and it
turns down what is decorative rather than removing what a module does.

Nothing here changes what a run computes. Every setting it touches is
about how the application is DRAWN.

It is chosen automatically, it can be overridden either way, and it says
what it turned down -- a mode that quietly makes the application different
is one nobody can debug.
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
    """Cores this process may actually use, not cores the machine has.

    `os.sched_getaffinity` where it exists: a container or a scheduler can
    pin the process to fewer than `cpu_count` reports, and pinning is
    exactly the situation this mode is for.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):        # pragma: no cover - not Linux
        return max(1, os.cpu_count() or 1)


def total_memory_gib() -> Optional[float]:
    """Physical memory in GiB, or None when it cannot be read."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        return (pages * size) / (1024 ** 3)
    except (AttributeError, ValueError, OSError):   # pragma: no cover
        return None


def override() -> Optional[bool]:
    """What the environment says, or None when it says nothing."""
    raw = str(os.environ.get(OVERRIDE_VARIABLE, "")).strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return None


def measure() -> Dict[str, object]:
    """What the decision is made from, so it can be reported and tested."""
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
    """Whether to run in laptop mode, and the sentence that says why.

    :returns: ``(on, why)``. The reason is always populated, including when
        the answer is no -- "spaCR decided not to" is a thing a user reports
        and somebody has to be able to check.
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
    """Each thing the mode changes, and what a user loses by it.

    ONLY DRAWING. Nothing here is read by a pipeline, so a run computes the
    same answer in either mode -- which is the promise that makes an
    automatic decision acceptable at all.
    """
    return (
        ("the ambient background animation",
         "the moving backdrop stops; every colour and control is unchanged"),
        ("the backdrop blur",
         "panels sit on a flat ground instead of a blurred one"),
    )


def describe() -> str:
    """The whole decision as one block, for the launch log."""
    on, why = wanted()
    lines = [why]
    if on:
        lines.append("turned down, and only the drawing:")
        lines.extend(f"  - {what}: {cost}" for what, cost in what_it_turns_down())
        lines.append("a run computes exactly the same answer either way.")
    return "\n".join(lines)


def apply(on: Optional[bool] = None) -> Dict[str, object]:
    """Turn the mode on or off. Returns what was decided and what changed.

    :param on: force the answer; ``None`` measures the machine.
    :returns: ``{"on", "why", "changed"}`` -- `changed` names each setting
        actually written, so a caller can say what happened rather than
        claim it.
    """
    if on is None:
        on, why = wanted()
    else:
        why = (f"laptop mode: {'on' if on else 'off'} because the caller "
               f"asked for it")

    changed: List[str] = []
    if on:
        try:
            from . import preferences

            if preferences.ambient_enabled():
                preferences.set_ambient_enabled(False)
                changed.append("ambient_enabled=False")
        except Exception:                                # noqa: BLE001
            pass
    return {"on": on, "why": why, "changed": changed}
