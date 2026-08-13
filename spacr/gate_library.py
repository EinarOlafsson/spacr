"""Named gating strategies, saved with the project rather than in a file dialog.

Instruction 31's "saveable filter sets". The Gate Editor could already write a
strategy to a path and read one back, but only through a file chooser, which
makes reuse a matter of remembering where you put it. A screen is gated the
same way over and over -- "live singlets", "infected cells", "the debris
filter I always apply" -- and that is a LIBRARY, not a file.

So a strategy has a NAME and lives under ``<project>/gates/<name>.json``. The
Gate Editor lists what is there, applies one by name, and saves the current
gates under a name.

Qt-free on purpose, like :mod:`spacr.filters`: the library is a directory of
JSON files, and everything here is testable without a display.

**A name is not a path.** ``save(project, "../../etc/passwd", gates)`` must not
write outside the project, and a name with a slash in it is a mistake rather
than a subdirectory. :func:`slugify` is the whole of that rule and every entry
point goes through it.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

#: Where a project keeps its saved strategies.
LIBRARY_DIRNAME = "gates"

#: What a saved strategy is called on disk.
SUFFIX = ".json"

#: Characters a name may keep. Everything else becomes a hyphen -- a name is
#: shown in a dropdown and used as a filename, and those two audiences agree
#: on very little else.
_SAFE = re.compile(r"[^A-Za-z0-9 _-]+")


class GateLibraryError(ValueError):
    """A strategy that cannot be saved or read, and why."""


def slugify(name: str) -> str:
    """The filename for ``name``, with no way out of the library directory.

    :raises GateLibraryError: a name that is empty once cleaned. Writing it
        would produce ``.json``, an invisible file the list would then show
        with no name.
    """
    cleaned = _SAFE.sub("-", str(name or "")).strip(" .-_")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        raise GateLibraryError(
            f"{name!r} has no characters that can be a strategy name")
    return cleaned


def library_dir(project: str) -> str:
    """The library directory for ``project``. Not created."""
    return os.path.join(str(project), LIBRARY_DIRNAME)


def path_for(project: str, name: str) -> str:
    """Where the strategy called ``name`` lives.

    Always inside the library directory: the name is slugified first, so a
    name carrying ``/`` or ``..`` cannot climb out of it.
    """
    return os.path.join(library_dir(project), slugify(name) + SUFFIX)


def list_strategies(project: str) -> List[str]:
    """Every saved strategy in ``project``, by name, sorted.

    An unreadable directory is an empty library rather than an error: a
    dropdown that cannot be filled is not a reason to refuse to open a screen.
    """
    directory = library_dir(project)
    try:
        entries = os.listdir(directory)
    except OSError:
        return []
    return sorted(e[:-len(SUFFIX)] for e in entries
                  if e.endswith(SUFFIX) and len(e) > len(SUFFIX))


def save(project: str, name: str, payload: Any) -> str:
    """Write ``payload`` as the strategy called ``name``.

    :param payload: whatever ``GateSet.to_json``-shaped structure the caller
        holds. Serialised here rather than accepting a pre-made string so a
        caller cannot store something that will not read back.
    :returns: the path written.
    :raises GateLibraryError: the name is unusable, or the payload will not
        serialise -- caught here rather than leaving a half-written file.
    """
    target = path_for(project, name)
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
    except OSError as exc:
        raise GateLibraryError(f"cannot create the gate library: {exc}") from exc
    try:
        text = json.dumps(payload, indent=2, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise GateLibraryError(
            f"that gating strategy cannot be saved: {exc}") from exc
    # Written whole and then moved, so an interrupted save leaves the previous
    # strategy intact rather than a truncated file the library still lists.
    temporary = target + ".part"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, target)
    except OSError as exc:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise GateLibraryError(f"cannot write {target}: {exc}") from exc
    return target


def load(project: str, name: str) -> Any:
    """Read the strategy called ``name``.

    :raises GateLibraryError: no such strategy, or the file is not readable
        JSON. Both name the strategy, because "expecting value: line 1" on
        its own tells a user nothing about which one to fix.
    """
    target = path_for(project, name)
    try:
        with open(target, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        known = list_strategies(project)
        raise GateLibraryError(
            f"no saved strategy called {name!r}"
            + (f"; this project has {', '.join(known)}" if known
               else "; this project has none")) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise GateLibraryError(
            f"the saved strategy {name!r} could not be read: {exc}") from exc


def delete(project: str, name: str) -> bool:
    """Remove the strategy called ``name``. ``False`` if there was none."""
    try:
        os.unlink(path_for(project, name))
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise GateLibraryError(f"cannot remove {name!r}: {exc}") from exc


def describe(project: str, name: str) -> Tuple[int, Optional[str]]:
    """``(gate count, error)`` for one saved strategy, without applying it.

    What a list needs to show next to a name. A strategy that will not read
    reports its error rather than a count, so a broken file is visible in the
    list instead of at the moment someone applies it.
    """
    try:
        payload = load(project, name)
    except GateLibraryError as exc:
        return 0, str(exc)
    if isinstance(payload, dict):
        gates = payload.get("gates")
        if isinstance(gates, (list, tuple)):
            return len(gates), None
        return len(payload), None
    if isinstance(payload, (list, tuple)):
        return len(payload), None
    return 0, f"{name!r} does not look like a gating strategy"
